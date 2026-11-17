"""Inspectable real image features, separate from evaluation ground truth."""
import time
import uuid

import numpy as np

from .. import data, retrieval
from . import encoders, service, store, pa1


def describe(vector, image_id, model, config, source, **extra):
    values=np.asarray(vector,dtype=np.float32)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Vector ảnh không hợp lệ')
    result=dict(id=uuid.uuid4().hex,created_at=data.now(),image_id=image_id,model=model,config=config,
                source=source,dimension=len(values),vector=values.tolist(),
                statistics=dict(min=float(values.min()),max=float(values.max()),mean=float(values.mean()),
                                std=float(values.std()),l2_norm=float(np.linalg.norm(values)),
                                zero_count=int(np.count_nonzero(values==0))),**extra)
    if model=='handcrafted':
        result['blocks']=pa1.blocks(values,config)
    data.save('image_features',result['id'],result)
    return result


def extract(image_id,config,job):
    job.update(0,1,'Đang mã hóa ảnh để xem vector đặc trưng')
    start=time.perf_counter()
    image=store.load(image_id)
    with encoders.manager.lock:
        cfg,loading_ms=service.prepare(config)
        vector,timing=encoders.manager.encode([image],cfg)
        job.check()
        result=describe(vector[0],image_id,cfg.model,cfg.model_dump(),'encoder',model_info=encoders.manager.info.copy(),
                        timing=dict(model_loading_ms=loading_ms,**timing,total_ms=(time.perf_counter()-start)*1000),
                        resources=retrieval.resources())
    job.update(1,1,'Đã lưu vector thực và thống kê')
    return result


def indexed(index_id,image_id):
    meta=data.get('image_indexes',index_id)
    if meta['status']!='ready' or image_id not in meta['images']:
        raise ValueError('Ảnh chưa có trong index đã sẵn sàng')
    client=retrieval.qclient()
    try:
        points=client.retrieve(collection_name=index_id,ids=[str(uuid.UUID(image_id))],with_vectors=True,with_payload=False)
    finally:
        client.close()
    if len(points)!=1:
        raise ValueError('Không tìm thấy vector trong Qdrant')
    return describe(points[0].vector,image_id,meta['config']['model'],meta['config'],'qdrant',index_id=index_id,
                    model_info=meta['model'])
