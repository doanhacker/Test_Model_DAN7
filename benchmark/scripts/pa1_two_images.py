"""PA1 Gallery capped at two images per place; reuse real existing vectors."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import uuid

import httpx
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NAMESPACE='danang-pa1-two-images'
DATASET='982441763d72185252745cdc'


def main(download):
    from backend import data,database,retrieval
    from qdrant_client import models as qm
    source=next(i for i in data.listing('image_indexes') if i.get('namespace')=='danang-pa1-live' and i['status']=='ready')
    config=source['config']
    signature=dict(encoder=config,namespace=NAMESPACE,algorithm_version='image-retrieval-v1')
    identity='img-'+data.digest(signature)[:28]
    try:
        meta=data.get('image_indexes',identity)
    except FileNotFoundError:
        meta=dict(source,id=identity,created_at=data.now(),images={},namespace=NAMESPACE,status='building',
                  max_images_per_place=2,embedding_ms=0.,preprocessing_ms=0.,source_index_id=source['id'],
                  vector_reuse_note='Existing PA1 vectors copied without re-encoding; same encoder/configuration.')
    # Existing entries stay selected on retries, preventing a changing subset.
    selected=dict(meta['images'])
    counts=Counter(a['place_id'] for a in selected.values())
    for asset in sorted(source['images'].values(),key=lambda a:(a['place_id'],a['created_at'],a['id'])):
        if asset['id'] not in selected and counts[asset['place_id']]<2:
            selected[asset['id']]=asset;counts[asset['place_id']]+=1
    pending=[key for key in selected if key not in meta['images']]
    client=retrieval.qclient()
    started=time.perf_counter()
    try:
        if not client.collection_exists(identity):
            client.create_collection(identity,vectors_config=qm.VectorParams(size=source['dimension'],distance=qm.Distance.COSINE))
            pending=list(selected)
        meta['status']='building';data.save('image_indexes',identity,meta)
        for offset in range(0,len(pending),64):
            batch=pending[offset:offset+64]
            points=client.retrieve(source['id'],ids=[str(uuid.UUID(key)) for key in batch],with_vectors=True,with_payload=True)
            if len(points)!=len(batch):raise ValueError('Source vector missing')
            if any(len(p.vector)!=source['dimension'] or not np.isfinite(p.vector).all() for p in points):
                raise ValueError('Invalid source vector')
            client.upsert(identity,points=[qm.PointStruct(id=p.id,vector=p.vector,payload=p.payload) for p in points],wait=True)
            for key in batch:meta['images'][key]=selected[key]
            data.save('image_indexes',identity,meta)
        meta.update(status='ready',indexed_count=client.count(identity,exact=True).count,
                    gallery_sha256=data.digest(sorted(meta['images'])),updated_at=data.now(),
                    vector_bytes=len(meta['images'])*source['dimension']*4,
                    last_indexing_ms=(time.perf_counter()-started)*1000,resources=retrieval.resources())
        if meta['indexed_count']!=len(meta['images']) or max(Counter(a['place_id'] for a in meta['images'].values()).values(),default=0)>2:
            raise ValueError('Two-image Gallery invariant violated')
        data.save('image_indexes',identity,meta)
    except BaseException as exc:
        meta.update(status='failed',error=str(exc));data.save('image_indexes',identity,meta);raise
    finally:
        client.close()
    report=dict(id='pa1-two-images',created_at=data.now(),status='ready',index_id=identity,indexed=len(meta['images']),
                places=len({a['place_id'] for a in meta['images'].values()}),max_images_per_place=2,errors=[])
    target=ROOT/'runtime'/'pa1-two-images.json'
    def save():
        report['updated_at']=data.now()
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        data.save('image_vectorizations',report['id'],report)
    save();print(json.dumps(report),flush=True)
    if not download:return
    with database.session(data.ROOT,data.db_path_override()) as conn:
        rows=conn.execute('''SELECT image_id,place_id FROM (SELECT image_id,place_id,ROW_NUMBER() OVER
                            (PARTITION BY place_id ORDER BY (image_content IS NOT NULL) DESC,is_cover DESC,ordinal,image_id) AS rn
                            FROM place_images WHERE dataset_id=?) ranked WHERE rn<=2 ORDER BY place_id,rn''',(DATASET,)).fetchall()
    report.update(status='running',source_total=len(rows),current=0)
    counts=Counter(a['place_id'] for a in meta['images'].values())
    known=set(meta['images'])
    batch=[]
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=90) as c:
        def flush():
            if not batch:return
            r=c.post('/api/ai/images/index-assets',json={'image_ids':list(batch),'config':{**config,'collection':NAMESPACE}})
            r.raise_for_status();job_id=r.json()['id'];report['job_id']=job_id;save()
            while True:
                j=c.get('/api/jobs/'+job_id);j.raise_for_status();job=j.json()
                if job['status'] not in ('queued','running'):
                    if job['status']!='completed':raise RuntimeError(job.get('error',job['status']))
                    report['indexed']=job['result']['total'];break
                time.sleep(2)
            batch.clear();save();time.sleep(2)
        try:
            for number,row in enumerate(rows,1):
                report['current']=number
                if counts[row['place_id']]>=2:
                    continue
                try:
                    try:
                        link=data.get('image_source_links',row['image_id']);asset=data.get('image_assets',link['image_id'])
                    except FileNotFoundError:
                        raw=c.get('/api/place-images/'+row['image_id']);raw.raise_for_status()
                        r=c.post('/api/ai/images/assets',files={'file':('source.jpg',raw.content,'image/jpeg')},data={'place_id':row['place_id'],'dataset_id':DATASET})
                        r.raise_for_status();asset=r.json()
                        data.save('image_source_links',row['image_id'],dict(id=row['image_id'],image_id=asset['id'],source_image_id=row['image_id'],dataset_id=DATASET,place_id=row['place_id'],created_at=data.now()))
                    if asset['id'] not in known:
                        known.add(asset['id']);batch.append(asset['id']);counts[row['place_id']]+=1
                except Exception as exc:
                    report['errors'].append({'source_image_id':row['image_id'],'place_id':row['place_id'],'error':str(exc)})
                if len(batch)>=32:
                    flush()
                if number%10==0:save()
            flush();report.update(status='completed',places=len(counts));save()
        except BaseException as exc:
            report.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download',action='store_true')
    main(parser.parse_args().download)
