import io
import json
import logging
import os

import pandas as pd
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from .. import data, jobs
from . import benchmark, encoders, service, store, features, pa1, importer, cnn

log=logging.getLogger(__name__)
router=APIRouter(prefix='/api/ai/images',tags=['Image-to-Image Search'])


def submit(kind,function):
    def task(job):
        try:
            return function(job)
        except Exception:
            log.exception('Image task failed: %s',kind)
            raise
        finally:
            # Release image model after each job; the shared worker prevents
            # concurrent indexing/search/benchmark/text loads.
            encoders.manager.unload()
    return jobs.submit('image_'+kind,task)


async def read_upload(file):
    raw=await file.read(store.MAX_BYTES+1)
    if len(raw)>store.MAX_BYTES:
        raise ValueError('Image exceeds upload byte limit')
    return raw


def parse_config(value):
    decoded=json.loads(value)
    if not isinstance(decoded,dict):
        raise ValueError('config must be a JSON object')
    return service.IndexConfig(**decoded)


@router.get('/models')
def models():
    return dict(models=encoders.MODELS,ablation=[encoders.GROUPS[:i] for i in range(1,6)],score='Cosine Similarity',normalization='L2')


@router.get('/indexes')
def indexes():
    return data.listing('image_indexes')


@router.get('/assets')
def assets():
    return data.listing('image_assets')


class SourceImport(BaseModel):
    dataset_id:str
    place_ids:list[str] | None=Field(default=None,min_length=1)
    limit:int=Field(default=100,ge=1,le=10000)
    all_images:bool=False


@router.post('/import-source')
def import_source(body:SourceImport):
    dataset=data.get('datasets',body.dataset_id)
    if dataset['kind']!='places':
        raise ValueError('Cần dataset địa điểm')
    if body.place_ids is not None and set(body.place_ids)-{p['place_id'] for p in dataset['records']}:
        raise ValueError('place_id không thuộc dataset')
    return submit('import',lambda job:importer.import_photos(body.dataset_id,body.place_ids,body.limit,body.all_images,job))


@router.get('/imports')
def source_imports():
    return data.listing('image_imports')


class FeatureRequest(BaseModel):
    image_id:str
    config:service.IndexConfig=Field(default_factory=service.IndexConfig)


@router.post('/features')
def extract_features(body:FeatureRequest):
    data.get('image_assets',body.image_id)
    return submit('features',lambda job:features.extract(body.image_id,body.config,job))


@router.get('/features')
def feature_history():
    return [{k:v for k,v in f.items() if k!='vector'} for f in data.listing('image_features')]


@router.get('/features/{key}')
def feature_detail(key:str):
    return data.get('image_features',key)


@router.get('/indexes/{index_id}/vectors/{image_id}')
def indexed_vector(index_id:str,image_id:str):
    return features.indexed(index_id,image_id)


@router.get('/assets/{image_id}/content')
def content(image_id:str):
    data.get('image_assets',image_id)
    return FileResponse(data.ROOT/'image_blobs'/(image_id+'.png'),media_type='image/png')


@router.get('/assets/{image_id}/pa1-preview')
def pa1_preview(image_id:str,kind:str='gray'):
    return Response(pa1.preview(store.load(image_id),kind),media_type='image/png',headers={'Cache-Control':'private, max-age=86400'})


@router.post('/assets')
async def upload(file:UploadFile=File(...),place_id:str=Form(''),name:str=Form(''),dataset_id:str=Form('')):
    return store.register(await read_upload(file),place_id.strip(),name,dataset_id or None)


@router.post('/index')
async def index(file:UploadFile=File(...),place_id:str=Form(...),config:str=Form('{}'),name:str=Form(''),dataset_id:str=Form('')):
    cfg=parse_config(config)
    if not place_id.strip():
        raise ValueError('place_id is required')
    asset=store.register(await read_upload(file),place_id.strip(),name,dataset_id or None)
    return submit('index',lambda job:service.index_images(cfg,[asset['id']],job))


@router.post('/index-batch')
async def index_batch(files:list[UploadFile]=File(...),metadata:str=Form(...),config:str=Form('{}')):
    cfg=parse_config(config)
    rows=json.loads(metadata)
    if not isinstance(rows,list) or len(rows)!=len(files) or not 1<=len(files)<=256:
        raise ValueError('metadata must have one object per file; max 256 images per batch')
    # Validate all images/metadata before committing any asset.
    raw_images=[]
    total_bytes=0
    for file,row in zip(files,rows):
        if not isinstance(row,dict) or not str(row.get('place_id','')).strip():
            raise ValueError('Each image requires place_id')
        raw=await read_upload(file)
        total_bytes+=len(raw)
        if total_bytes>int(os.getenv('IMAGE_MAX_BATCH_MB','100'))*1024*1024:
            raise ValueError('Batch exceeds IMAGE_MAX_BATCH_MB; split into smaller batches')
        store.decode(raw)
        raw_images.append(raw)
    assets=[store.register(raw,str(row['place_id']).strip(),row.get('name',''),row.get('dataset_id') or None) for raw,row in zip(raw_images,rows)]
    return submit('index_batch',lambda job:service.index_images(cfg,list(dict.fromkeys(a['id'] for a in assets)),job))


class ExistingIndex(BaseModel):
    image_ids:list[str]=Field(min_length=1)
    config:service.IndexConfig=Field(default_factory=service.IndexConfig)


class CNNRequest(BaseModel):
    source_index_id:str
    batch_size:int=Field(default=2,ge=1,le=16)


@router.post('/cnn-indexes')
def cnn_indexes(body:CNNRequest):
    source=data.get('image_indexes',body.source_index_id)
    selected=cnn.select_gallery(source)
    return submit('cnn',lambda job:cnn.build(source,selected,body.batch_size,job))


@router.post('/index-assets')
def index_assets(body:ExistingIndex):
    return submit('index_assets',lambda job:service.index_images(body.config,body.image_ids,job))


@router.post('/ablation')
def index_pa1_ablation(body:ExistingIndex):
    if body.config.model!='handcrafted':
        raise ValueError('Ablation này chỉ áp dụng PA1')
    if len(body.image_ids)!=len(set(body.image_ids)):
        raise ValueError('Gallery có image_id trùng')
    for key in body.image_ids:
        if not data.get('image_assets',key)['place_id']:
            raise ValueError('Gallery cần ảnh liên kết place_id')
    def task(job):
        results=[]
        total=len(body.image_ids)*5
        for stage in range(5):
            class Progress:
                def check(self):
                    job.check()
                def update(self,current,subtotal,message):
                    job.update(stage*len(body.image_ids)+current,total,f'PA1 ablation {stage+1}/5 · {message}')
            cfg=body.config.model_copy(update={'groups':encoders.GROUPS[:stage+1]})
            result=service.index_images(cfg,body.image_ids,Progress())
            results.append(dict(index_id=result['index_id'],added=result['added'],total=result['total'],groups=cfg.groups))
            job.update((stage+1)*len(body.image_ids),total,f'Hoàn tất PA1 ablation {stage+1}/5')
        return {'indexes':results}
    return submit('ablation',task)


@router.post('/search')
async def search(file:UploadFile=File(...),index_id:str=Form(...),top_k:int=Form(10,ge=1,le=1000),deduplicate:bool=Form(True),exact:bool=Form(True)):
    raw=await read_upload(file)
    store.decode(raw)
    return submit('search',lambda job:service.search_bytes(index_id,raw,top_k,deduplicate,exact))


@router.delete('/{image_id}')
def delete(image_id:str,index_id:str):
    return submit('delete',lambda job:service.delete_image(index_id,image_id))


@router.post('/datasets')
def create_dataset(body:benchmark.Dataset):
    return benchmark.create_dataset(body)


@router.get('/datasets')
def datasets():
    return data.listing('image_datasets')


@router.post('/benchmarks')
def run(body:benchmark.Config):
    benchmark.validate(body)
    return submit('benchmark',lambda job:benchmark.run(body,job))


@router.get('/runs')
def runs():
    return data.ram_only([{k:v for k,v in r.items() if k not in ('samples','detail','cold_start')} for r in data.listing('image_runs')])


@router.get('/runs/{key}')
def get_run(key:str):
    return data.ram_only(data.get('image_runs',key))


@router.get('/runs/{key}/export')
def export(key:str,format:str='json'):
    result=data.ram_only(data.get('image_runs',key))
    if format=='json':
        content=json.dumps(result,ensure_ascii=False,indent=2).encode()
        mime='application/json'
    elif format=='csv':
        import zipfile
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            for field in ('summary','detail'):
                archive.writestr(field+'.csv',pd.DataFrame(result[field]).to_csv(index=False).encode('utf-8-sig'))
            archive.writestr('provenance-and-rankings.json',json.dumps(result,ensure_ascii=False,indent=2))
        content,mime,format=buffer.getvalue(),'application/zip','zip'
    else:
        raise ValueError('Only JSON or CSV ZIP')
    return Response(content,media_type=mime,headers={'Content-Disposition':f'attachment; filename="image-benchmark-{key}.{format}"'})
