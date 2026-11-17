import importlib.metadata
import threading
import time
import uuid
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .. import data, retrieval
from ..metrics import evaluate_query, latency_summary
from . import service, store, encoders


class Query(BaseModel):
    query_id: str = Field(min_length=1)
    image_id: str
    exact_place_ids: list[str] = Field(default_factory=list)


class Judgment(BaseModel):
    query_id: str
    target_id: str
    relevance: int = Field(ge=0,le=3,strict=True)
    reason:str=''
    review_status:Literal['confirmed']='confirmed'


class Dataset(BaseModel):
    name: str = 'Image benchmark dataset'
    gallery_image_ids: list[str] = Field(min_length=1)
    queries: list[Query] = Field(min_length=1)
    visual_target: Literal['place','image'] = 'place'
    judgments: list[Judgment] = Field(default_factory=list)


def create_dataset(body):
    if len(set(body.gallery_image_ids))!=len(body.gallery_image_ids):
        raise ValueError('Duplicate gallery image_id')
    gallery=[data.get('image_assets',i) for i in body.gallery_image_ids]
    if any(not i['place_id'] for i in gallery):
        raise ValueError('Gallery image needs place_id')
    pids={i['place_id'] for i in gallery}
    ids={q.query_id for q in body.queries}
    if len(ids)!=len(body.queries) or len({q.image_id for q in body.queries})!=len(body.queries):
        raise ValueError('Duplicate query_id/image_id')
    gallery_hashes={i['pixel_sha256'] for i in gallery}
    query_assets=[]
    for query in body.queries:
        asset=data.get('image_assets',query.image_id)
        query_assets.append(asset)
        if asset['pixel_sha256'] in gallery_hashes:
            raise ValueError('Query duplicates Gallery pixels: leakage is not allowed')
        if set(query.exact_place_ids)-pids or len(set(query.exact_place_ids))!=len(query.exact_place_ids):
            raise ValueError('Unknown/duplicate exact place ground truth')
    targets=pids if body.visual_target=='place' else set(body.gallery_image_ids)
    pairs=set()
    for row in body.judgments:
        pair=(row.query_id,row.target_id)
        if pair in pairs or row.query_id not in ids or row.target_id not in targets:
            raise ValueError('Invalid or duplicate visual judgment')
        pairs.add(pair)
    value=body.model_dump()
    key=data.digest(value)[:24]
    try:
        return data.get('image_datasets',key)
    except FileNotFoundError:
        pass
    value.update(id=key,created_at=data.now(),sha256=data.digest(value),gallery_assets=gallery,query_assets=query_assets)
    data.save('image_datasets',key,value)
    return value


class Config(BaseModel):
    dataset_id: str
    index_ids: list[str] = Field(min_length=1)
    objective: Literal['exact','visual','both'] = 'both'
    repeats: int = Field(default=3,ge=1,le=100)
    warmup: int = Field(default=1,ge=0,le=20)
    threshold: int = Field(default=1,ge=1,le=3)
    official: bool = True
    query_ids: list[str] | None = Field(default=None,min_length=1)


def validate(config):
    ds=data.get('image_datasets',config.dataset_id)
    if config.query_ids is not None:
        selected=set(config.query_ids)
        if len(selected)!=len(config.query_ids) or selected-{q['query_id'] for q in ds['queries']}:
            raise ValueError('Truy vấn test không tồn tại hoặc bị chọn trùng')
        ds={**ds,'queries':[q for q in ds['queries'] if q['query_id'] in selected],
            'judgments':[j for j in ds['judgments'] if j['query_id'] in selected]}
    indexes=[data.get('image_indexes',key) for key in config.index_ids]
    if len(set(config.index_ids))!=len(indexes):
        raise ValueError('Duplicate index')
    if any(i['status']!='ready' or set(i['images'])!=set(ds['gallery_image_ids']) for i in indexes):
        raise ValueError('All indexes must be ready and contain exactly the same Gallery snapshot')
    targets={a['place_id'] for a in ds['gallery_assets']} if ds['visual_target']=='place' else set(ds['gallery_image_ids'])
    labels={q['query_id']:{} for q in ds['queries']}
    for row in ds['judgments']:
        labels[row['query_id']][row['target_id']]=row['relevance']
    if config.objective in ('exact','both') and any(not q['exact_place_ids'] for q in ds['queries']):
        raise ValueError('Exact retrieval requires explicit exact_place_ids for every query')
    if config.objective in ('visual','both') and config.official and any(set(labels[q['query_id']])!=targets for q in ds['queries']):
        missing=sum(len(targets-set(labels[q['query_id']])) for q in ds['queries'])
        raise ValueError(f'Visual chính thức còn thiếu {missing} nhãn query/target đã xác nhận. Nhãn trống không phải 0; hãy gán thêm hoặc chọn chế độ thăm dò.')
    return ds,indexes,targets,labels


def run(config,job):
    ds,indexes,targets,labels=validate(config)
    objectives=['exact','visual'] if config.objective=='both' else [config.objective]
    key=uuid.uuid4().hex
    versions=retrieval.environment()
    for name in ('torchvision','transformers','Pillow','opencv-python','scikit-image'):
        try:
            versions['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions['versions'][name]=None
    result=dict(id=key,created_at=data.now(),status='running',config=config.model_dump(),dataset=ds,indexes=indexes,
                environment=versions,summary=[],detail=[],samples=[],cold_start=[])
    peak=retrieval.resources()
    stop=threading.Event()
    def monitor():
        while not stop.wait(.1):
            current=retrieval.resources()
            for field in peak:
                if current[field] is not None:
                    peak[field]=max(peak[field] or 0,current[field])
    thread=threading.Thread(target=monitor,daemon=True)
    thread.start()
    total=len(indexes)*len(ds['queries'])*config.repeats*len(objectives)
    done=0
    data.save('image_runs',key,result)
    try:
        for index in indexes:
            encoders.manager.unload()
            first=store.load(ds['queries'][0]['image_id'])
            cold=service.retrieve(index['id'],first,10,True,True)
            result['cold_start'].append(cold)
            for _ in range(config.warmup):
                job.check()
                service.retrieve(index['id'],first,10,True,True)
            for objective in objectives:
                for repeat in range(config.repeats):
                    for query in ds['queries']:
                        job.check()
                        start=time.perf_counter()
                        image=store.load(query['image_id'])
                        decode_ms=(time.perf_counter()-start)*1000
                        dedup=objective=='exact' or ds['visual_target']=='place'
                        answer=service.retrieve(index['id'],image,10,dedup,True)
                        answer['timing']['decode_ms']=decode_ms
                        answer['timing']['total_ms']+=decode_ms
                        ranking=[r['place_id' if dedup else 'image_id'] for r in answer['results']]
                        if objective=='exact':
                            corpus={a['place_id'] for a in ds['gallery_assets']}
                            # Identity labels are defined ONLY for the exact-place task.
                            exact_labels={p:int(p in query['exact_place_ids']) for p in corpus}
                            metrics={'recall@1':evaluate_query(ranking,exact_labels,corpus,k=1)['recall'],
                                     'recall@5':evaluate_query(ranking,exact_labels,corpus,k=5)['recall'],
                                     'mrr@10':evaluate_query(ranking,exact_labels,corpus,k=10)['mrr'], 'status':'official'}
                        else:
                            values=evaluate_query(ranking,labels[query['query_id']],targets,k=10,threshold=config.threshold,official=config.official)
                            metrics={'ndcg@10':values['ndcg'],'mrr@10':values['mrr'],'precision@10':values['precision'],'recall@10':values['recall'],'status':values['status'],'coverage':values['coverage'],
                                     'missing_top_k':values['missing_top_k'],'missing_corpus':values['missing_corpus'],
                                     'evaluation_trace':values['trace'],'known_relevant':values['known_relevant'],
                                     'hits_at_k':values['hits_at_k'],'first_relevant_rank':values['first_relevant_rank'],
                                     'dcg':values['dcg'],'idcg':values['idcg']}
                        record=dict(index_id=index['id'],model=index['config']['model'],objective=objective,repeat=repeat+1,query_id=query['query_id'],**metrics,**answer['timing'],**answer['resources'])
                        result['detail'].append(record)
                        result['samples'].append(dict(query_id=query['query_id'],objective=objective,repeat=repeat+1,**answer))
                        done+=1
                        job.update(done,total,f"{record['model']} · {objective} · {query['query_id']}")
                rows=[r for r in result['detail'] if r['index_id']==index['id'] and r['objective']==objective]
                summary=dict(index_id=index['id'],model=index['config']['model'],objective=objective,samples=len(rows),
                             dimension=index['dimension'],vector_bytes=index['vector_bytes'],model_loading_ms=cold['timing']['model_loading_ms'],
                             **latency_summary([r['total_ms'] for r in rows]))
                for field in (['recall@1','recall@5','mrr@10'] if objective=='exact' else ['ndcg@10','mrr@10','precision@10','recall@10'])+['embedding_ms','preprocessing_ms','search_ms','ram_mb']:
                    values=[r[field] for r in rows if r.get(field) is not None]
                    summary[field]=float(np.mean(values)) if values else None
                    summary[field+'_samples']=len(values)
                result['summary'].append(summary)
                data.save('image_runs',key,result)
        result['status']='completed'
        return {'image_run_id':key}
    except BaseException as exc:
        result.update(status='cancelled' if job.cancel.is_set() else 'failed',error=str(exc))
        raise
    finally:
        stop.set()
        thread.join(timeout=1)
        result.update(peak_resources=peak,finished_at=data.now())
        data.save('image_runs',key,result)
