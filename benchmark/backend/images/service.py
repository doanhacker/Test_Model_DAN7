import time
import uuid
from typing import Literal

import numpy as np
from pydantic import Field
from qdrant_client import models as qm

from .. import data, retrieval
from . import encoders, store, pa1


class IndexConfig(encoders.EncoderConfig):
    collection: str = Field(default='tourism', pattern=r'^[a-zA-Z0-9_-]{1,40}$')


def encoder_config(config):
    return encoders.EncoderConfig(**config.model_dump())


def prepare(config):
    # Shared job queue serializes image and text tasks. Drop the text model first.
    if retrieval.manager.model is not None:
        retrieval.manager.unload()
    cfg = encoder_config(config)
    loading_ms = encoders.manager.load(cfg)
    resolved = encoders.manager.info
    if encoders.MODELS[cfg.model]['family'] in ('PA3','PA4'):
        cfg = cfg.model_copy(update={'revision': resolved['revision'], 'device': resolved['device']})
    else:
        cfg = cfg.model_copy(update={'device': resolved['device'],'revision':resolved['revision']})
    encoders.manager.key = cfg.model_dump_json()
    return cfg, loading_ms


def index_images(config, image_ids, job):
    assets = [data.get('image_assets',key) for key in image_ids]
    if not assets or any(not a['place_id'] for a in assets):
        raise ValueError('Gallery cần ít nhất một ảnh và place_id cho mỗi ảnh')
    if len(image_ids)!=len(set(image_ids)):
        raise ValueError('Batch chứa image_id trùng')
    start = time.perf_counter()
    with encoders.manager.lock:
        cfg, loading_ms = prepare(config)
        signature = dict(encoder=cfg.model_dump(),namespace=config.collection,algorithm_version='image-retrieval-v1')
        key = 'img-'+data.digest(signature)[:28]
        client = retrieval.qclient()
        try:
            try:
                meta = data.get('image_indexes',key)
            except FileNotFoundError:
                meta = dict(id=key,created_at=data.now(),config=cfg.model_dump(),namespace=config.collection,
                            model=encoders.manager.info.copy(),images={},status='building',dimension=None,
                            embedding_ms=0.,preprocessing_ms=0.,model_loading_ms=loading_ms,
                            environment=retrieval.environment())
            old_ids = set(meta['images'])
            exists=client.collection_exists(key)
            physical_count=client.count(key,exact=True).count if exists else 0
            if physical_count!=len(old_ids):
                # Recover missing points from immutable assets rather than mark a
                # disappeared/partial physical collection as ready on replay.
                pending=list({**meta['images'],**{a['id']:a for a in assets}}.values())
            else:
                pending = [a for a in assets if a['id'] not in old_ids]
            # Persist intent before upserts. Failed/partial collections cannot be searched.
            meta['status']='building'
            data.save('image_indexes',key,meta)
            for offset in range(0,len(pending),cfg.batch_size):
                job.check()
                batch = pending[offset:offset+cfg.batch_size]
                vectors, timing = encoders.manager.encode([store.load(a['id']) for a in batch],cfg)
                if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors,axis=1)<1e-8):
                    raise ValueError('Encoder sinh vector rỗng/không hữu hạn')
                if meta['dimension'] is None:
                    meta['dimension']=int(vectors.shape[1])
                elif meta['dimension']!=int(vectors.shape[1]):
                    raise ValueError('Embedding dimension changed for existing image index')
                # Creation may have failed on an earlier attempt after dimension
                # was persisted. Retry must still create the missing collection.
                if not client.collection_exists(key):
                    client.create_collection(key,vectors_config=qm.VectorParams(size=meta['dimension'],distance=qm.Distance.COSINE))
                points=[qm.PointStruct(id=str(uuid.UUID(a['id'])),vector=v.tolist(),payload={'image_id':a['id'],'place_id':a['place_id'],'name':a['name']}) for a,v in zip(batch,vectors)]
                client.upsert(key,points=points,wait=True)
                for asset in batch:
                    meta['images'][asset['id']]=asset
                meta['embedding_ms']+=timing['embedding_ms']
                meta['preprocessing_ms']+=timing['preprocessing_ms']
                data.save('image_indexes',key,meta)
                job.update(min(offset+cfg.batch_size,len(pending)),len(pending),'Đã index ảnh '+cfg.model)
            job.check()
            meta['indexed_count']=client.count(key,exact=True).count
            if meta['indexed_count']!=len(meta['images']):
                raise ValueError('Số vector Qdrant không khớp Gallery; cần kiểm tra / chạy lại indexing')
            meta.pop('error',None)
            meta.update(status='ready',updated_at=data.now(),gallery_sha256=data.digest(sorted(meta['images'])),
                        vector_bytes=len(meta['images'])*(meta['dimension'] or 0)*4,
                        storage_note='float32 vector bytes only; excludes Qdrant HNSW/payload/replicas',
                        last_indexing_ms=(time.perf_counter()-start)*1000,resources=retrieval.resources())
            data.save('image_indexes',key,meta)
            return dict(index_id=key,added=len(pending),total=len(meta['images']),metadata=meta)
        except Exception as exc:
            if 'meta' in locals():
                meta.update(status='failed',error=str(exc))
                data.save('image_indexes',key,meta)
            raise
        finally:
            client.close()


def retrieve(index_id, image, k=10, deduplicate=True, exact=True, target='place'):
    total = time.perf_counter()
    meta = data.get('image_indexes',index_id)
    if meta['status']!='ready' or not meta['images']:
        raise ValueError('Image index chưa sẵn sàng hoặc không có ảnh')
    cfg=encoders.EncoderConfig(**meta['config'])
    with encoders.manager.lock:
        cfg, loading_ms=prepare(cfg)
        vectors,timing=encoders.manager.encode([image],cfg)
        client=retrieval.qclient()
        try:
            start=time.perf_counter()
            # Fetch every image when collapsing places, so duplicate-heavy places
            # cannot hide the real Top-K unique places. This is intentionally exact.
            limit=len(meta['images']) if deduplicate else min(k,len(meta['images']))
            points=client.query_points(index_id,query=vectors[0].tolist(),limit=limit,with_payload=True,
                                       search_params=qm.SearchParams(exact=exact)).points
            search_ms=(time.perf_counter()-start)*1000
        finally:
            client.close()
    explanation_ms=0.
    seen=set()
    results=[]
    for p in points:
        payload=p.payload
        if payload['image_id'] not in meta['images']:
            continue
        entity=payload['place_id'] if deduplicate else payload['image_id']
        if entity in seen:
            continue
        seen.add(entity)
        results.append(dict(**payload,rank=len(results)+1,score=float(p.score),image_url='/api/ai/images/assets/'+payload['image_id']+'/content'))
        if len(results)>=k:
            break
    if cfg.model=='handcrafted' and results:
        start=time.perf_counter()
        client=retrieval.qclient()
        try:
            stored=client.retrieve(index_id,ids=[str(uuid.UUID(r['image_id'])) for r in results],with_vectors=True,with_payload=False)
        finally:
            client.close()
        lookup={str(p.id).replace('-',''):p.vector for p in stored}
        for result in results:
            if result['image_id'] in lookup:
                result['feature_scores']=pa1.explain(vectors[0],lookup[result['image_id']],cfg.model_dump())
        explanation_ms=(time.perf_counter()-start)*1000
    # Resolve original place fields, preserving the image/place association.
    from .. import database
    datasets={meta['images'][r['image_id']].get('dataset_id') for r in results}
    for dataset_id in datasets-{None,''}:
        ids=[r['place_id'] for r in results if meta['images'][r['image_id']].get('dataset_id')==dataset_id]
        with database.session(data.ROOT,data.db_path_override()) as conn:
            places=conn.execute('SELECT place_id,address,category,description FROM places WHERE dataset_id=? AND place_id=ANY(?)',(dataset_id,ids)).fetchall()
        lookup={p['place_id']:dict(p) for p in places}
        for result in results:
            if meta['images'][result['image_id']].get('dataset_id')==dataset_id:
                result.update(lookup.get(result['place_id'],{}))
    return dict(index_id=index_id,model=cfg.model,score_type='Cosine Similarity',deduplicate=deduplicate,exact=exact,
                results=results,timing=dict(model_loading_ms=loading_ms,**timing,search_ms=search_ms,explanation_ms=explanation_ms,total_ms=(time.perf_counter()-total)*1000),resources=retrieval.resources())


def search_bytes(index_id,raw,k=10,deduplicate=True,exact=True):
    start=time.perf_counter()
    image=store.decode(raw)
    decode_ms=(time.perf_counter()-start)*1000
    result=retrieve(index_id,image,k,deduplicate,exact)
    result['timing']['decode_ms']=decode_ms
    result['timing']['total_ms']+=decode_ms
    return result


def delete_image(index_id,image_id):
    meta=data.get('image_indexes',index_id)
    data.get('image_assets',image_id)
    client=retrieval.qclient()
    try:
        prior_status=meta['status']
        meta['status']='building'
        data.save('image_indexes',index_id,meta)
        client.delete(index_id,points_selector=qm.PointIdsList(points=[str(uuid.UUID(image_id))]),wait=True)
        meta['images'].pop(image_id,None)
        meta.update(status=prior_status if meta['images'] else 'empty',indexed_count=client.count(index_id,exact=True).count,
                    gallery_sha256=data.digest(sorted(meta['images'])),updated_at=data.now(),vector_bytes=len(meta['images'])*(meta['dimension'] or 0)*4)
        data.save('image_indexes',index_id,meta)
    finally:
        client.close()
    return {'image_id':image_id,'index_id':index_id,'deleted':True,'note':'Vector removed; immutable image asset retained for provenance.'}
