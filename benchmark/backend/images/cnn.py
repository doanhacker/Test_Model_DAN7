"""Frozen PA1/CNN comparison on an identical, bounded Gallery snapshot."""
from collections import Counter
import time
import uuid

from qdrant_client import models as qm

from .. import data,retrieval
from . import encoders,service

MODELS=['mobilenet_v3_large','resnet50','efficientnet_b0']


def select_gallery(source):
    if source['status']!='ready' or source['config']['model']!='handcrafted':
        raise ValueError('Chọn image index PA1 đã sẵn sàng làm Gallery chuẩn')
    counts=Counter()
    selected=[]
    for identity,asset in sorted(source['images'].items()):
        if counts[asset['place_id']]<2:
            selected.append(identity);counts[asset['place_id']]+=1
    if not selected:
        raise ValueError('Gallery chuẩn chưa có ảnh')
    return selected


def snapshot_baseline(source,image_ids,namespace,job):
    signature=dict(encoder=source['config'],namespace=namespace,algorithm_version='image-retrieval-v1')
    identity='img-'+data.digest(signature)[:28]
    meta=dict(source,id=identity,created_at=data.now(),updated_at=data.now(),images={},namespace=namespace,
              status='building',max_images_per_place=2,source_index_id=source['id'],embedding_ms=0.,preprocessing_ms=0.,
              vector_reuse_note='Frozen PA1 comparison snapshot; copied real source vectors without re-encoding')
    client=retrieval.qclient();started=time.perf_counter()
    try:
        data.save('image_indexes',identity,meta)
        if not client.collection_exists(identity):
            client.create_collection(identity,vectors_config=qm.VectorParams(size=source['dimension'],distance=qm.Distance.COSINE))
        for offset in range(0,len(image_ids),64):
            job.check();batch=image_ids[offset:offset+64]
            points=client.retrieve(source['id'],ids=[str(uuid.UUID(i)) for i in batch],with_vectors=True,with_payload=True)
            if len(points)!=len(batch):raise ValueError('Gallery gốc thiếu vector PA1')
            client.upsert(identity,points=[qm.PointStruct(id=p.id,vector=p.vector,payload=p.payload) for p in points],wait=True)
            meta['images'].update({i:source['images'][i] for i in batch})
        meta['indexed_count']=client.count(identity,exact=True).count
        if meta['indexed_count']!=len(image_ids):raise ValueError('Số vector snapshot không khớp Gallery')
        meta.update(status='ready',gallery_sha256=data.digest(sorted(image_ids)),vector_bytes=len(image_ids)*source['dimension']*4,
                    last_indexing_ms=(time.perf_counter()-started)*1000,resources=retrieval.resources())
        meta.pop('error',None);data.save('image_indexes',identity,meta);return meta
    except BaseException as exc:
        meta.update(status='failed',error=str(exc));data.save('image_indexes',identity,meta);raise
    finally:
        client.close()


def build(source,image_ids,batch_size,job):
    namespace='cnn-'+data.digest(sorted(image_ids))[:16]
    count=len(image_ids)
    job.update(0,count*3,'Chốt Gallery tối đa 2 ảnh/địa điểm và sao chép baseline PA1')
    baseline=snapshot_baseline(source,image_ids,namespace,job)
    results=[]
    for stage,model in enumerate(MODELS):
        job.check();job.update(stage*count,count*3,f'PA2 · Đang tải / nạp {model} trên CPU')
        class Progress:
            def check(self):job.check()
            def update(self,current,total,message):job.update(stage*count+current,count*3,model+' · '+message)
        cfg=service.IndexConfig(model=model,device='cpu',batch_size=batch_size,collection=namespace)
        result=service.index_images(cfg,image_ids,Progress())
        meta=result['metadata'];meta.update(max_images_per_place=2,comparison_gallery_sha256=data.digest(sorted(image_ids)),baseline_index_id=baseline['id'])
        data.save('image_indexes',meta['id'],meta)
        results.append(dict(model=model,index_id=meta['id'],dimension=meta['dimension'],count=result['total']))
        encoders.manager.unload()
    return dict(baseline_index_id=baseline['id'],gallery_images=count,gallery_sha256=data.digest(sorted(image_ids)),indexes=results)
