import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pandas as pd
import psycopg
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from . import data, jobs, retrieval, runner, real_data, database, place_media, query_evaluation
from .images.api import router as image_router


@asynccontextmanager
async def lifespan(app):
    database.initialize(data.ROOT,data.db_path_override())
    jobs.recover()
    yield


app = FastAPI(title='AI Search Benchmark · PA1 & PA2', lifespan=lifespan)
app.include_router(image_router)
app.include_router(place_media.router)


@app.exception_handler(ValueError)
async def invalid(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={'detail': str(exc)})


@app.exception_handler(FileNotFoundError)
async def missing(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=404, content={'detail': 'Không tìm thấy phiên bản/ID'})


@app.exception_handler(httpx.HTTPError)
async def upstream(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503, content={'detail': 'Dịch vụ tìm kiếm chưa sẵn sàng: ' + str(exc)})


@app.exception_handler(psycopg.OperationalError)
async def postgres_unavailable(request,exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=503,content={'detail':'Không kết nối được PostgreSQL. Kiểm tra service postgres và DATABASE_URL trong .env.'})


@app.get('/api/health')
def health():
    services = {}
    for name, url in [('elasticsearch', retrieval.ES_URL), ('qdrant', retrieval.QDRANT_URL)]:
        try:
            with httpx.Client(timeout=2) as client:
                r = client.get(url)
                services[name] = {'ready': r.is_success, 'status': r.status_code}
        except httpx.HTTPError as exc:
            services[name] = {'ready': False, 'error': str(exc)}
    return dict(services=services, resources=retrieval.resources(), loaded_model=retrieval.manager.info,database=data.database_info())


@app.get('/api/database/info')
def database_info():
    return data.database_info()


@app.get('/api/database/places')
def database_places(dataset_id:str,page:int=1,page_size:int=50,search:str=''):
    if page<1 or not 1<=page_size<=200:
        raise ValueError('page >= 1; page_size 1–200')
    where='dataset_id=?'
    args=[dataset_id]
    if search:
        where+=' AND (name ILIKE ? OR address ILIKE ? OR description ILIKE ?)'
        args += ['%'+search+'%']*3
    with database.session(data.ROOT,data.db_path_override()) as conn:
        count=conn.execute('SELECT COUNT(*) FROM places WHERE '+where,args).fetchone()[0]
        rows=conn.execute('SELECT * FROM places WHERE '+where+' ORDER BY ordinal LIMIT ? OFFSET ?',args+[page_size,(page-1)*page_size]).fetchall()
    records=[]
    for row in rows:
        value=dict(row)
        value['tags']=json.loads(value.pop('tags_json'))
        value['amenities']=json.loads(value.pop('amenities_json'))
        records.append(value)
    return dict(total=count,page=page,page_size=page_size,records=place_media.attach(dataset_id,records))


@app.get('/api/database/history')
def database_history(dataset_id:str='',method:str='',page:int=1,page_size:int=50):
    if page<1 or not 1<=page_size<=200:
        raise ValueError('page >= 1; page_size 1–200')
    where='1=1'
    args=[]
    for field,value in [('r.dataset_id',dataset_id),('m.method',method)]:
        if value:
            where+=' AND '+field+'=?'
            args.append(value)
    base=' FROM benchmark_runs r LEFT JOIN run_metrics m ON r.run_kind=m.run_kind AND r.id=m.run_id WHERE '+where
    with database.session(data.ROOT,data.db_path_override()) as conn:
        count=conn.execute('SELECT COUNT(*)'+base,args).fetchone()[0]
        rows=conn.execute('SELECT r.run_kind,r.id AS run_id,r.dataset_id,r.status,r.created_at,r.finished_at,m.index_id,m.method,m.objective,m.k,m.query_count,m.sample_count,m.recall_at_k,m.mrr_at_k,m.ndcg_at_k,m.precision_at_k,m.average_ms,m.p50_ms,m.p95_ms,m.p99_ms,m.ram_mb'+base+' ORDER BY r.created_at DESC LIMIT ? OFFSET ?',args+[page_size,(page-1)*page_size]).fetchall()
    return dict(total=count,page=page,page_size=page_size,records=[dict(r) for r in rows])


@app.get('/api/database/backup')
def database_backup():
    import uuid
    from starlette.background import BackgroundTask
    folder=data.ROOT/'database_backups'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/(uuid.uuid4().hex+'.sql')
    database.backup(data.ROOT,target,data.db_path_override())
    return FileResponse(target,filename='benchmark-postgres-backup.sql',media_type='application/sql',background=BackgroundTask(target.unlink))


@app.post('/api/database/runs/{run_id}/rerun')
def rerun_saved_configuration(run_id:str):
    saved=data.get('runs',run_id)
    if saved.get('evaluation_only'):
        original=saved['config']
        evaluation=query_evaluation.Config(job_id=original['source_job_id'],query_id=original.get('query_id',saved['query_set'][0]['query_id']),
                                           qrels_id=original['qrels_id'],k=original['k'],threshold=original['threshold'],official=original['official'])
        return jobs.submit('query_evaluation',lambda job:{'run_id':query_evaluation.calculate(evaluation)['id']})
    config=runner.BenchmarkConfig(**saved['config'])
    runner.validate_config(config)
    return jobs.submit('benchmark',lambda job:runner.run(config,job))


@app.get('/api/models')
def models():
    return retrieval.MODELS


@app.get('/api/datasets')
def datasets():
    return [{k: v for k, v in d.items() if k != 'records'} for d in data.listing('datasets')]


@app.get('/api/datasets/{key}')
def dataset(key: str):
    return data.get('datasets', key)


@app.post('/api/preview')
async def preview(file: UploadFile = File(...)):
    frame = data.parse(await file.read(), file.filename or '')
    return dict(columns=frame.columns.tolist(), count=len(frame), records=frame.head(20).to_dict('records'))


@app.post('/api/import')
async def upload(file: UploadFile = File(...), kind: str = Form(...), mapping: str = Form('{}'), dataset_id: str = Form(''), queries_id: str = Form('')):
    raw = await file.read()
    frame = data.parse(raw, file.filename or '')
    return data.import_records(kind, frame.to_dict('records'), file.filename, dataset_id or None, queries_id or None, raw, json.loads(mapping))


class Labels(BaseModel):
    dataset_id: str
    queries_id: str
    records: list[dict]
    name: str = 'Gán nhãn thủ công'


@app.post('/api/labels')
def labels(body: Labels):
    return data.import_records('qrels', body.records, body.name, body.dataset_id, body.queries_id)


@app.post('/api/real-data/import-excel')
async def import_tourism(file: UploadFile=File(...),city: str=Form('Đà Nẵng')):
    return real_data.import_excel(await file.read(),file.filename or 'places.xlsx',city)


class QueryGeneration(BaseModel):
    dataset_id: str


@app.post('/api/real-data/queries')
def generate_queries(body: QueryGeneration):
    return real_data.generate_queries(body.dataset_id)


class PoolGeneration(BaseModel):
    index_id: str
    queries_id: str


@app.post('/api/real-data/pool')
def generate_pool(body: PoolGeneration):
    return jobs.submit('bm25_pool',lambda job:real_data.create_pool(body.index_id,body.queries_id,job))


@app.get('/api/real-data/coverage')
def label_coverage(dataset_id:str,queries_id:str,qrels_id:str):
    return real_data.coverage(dataset_id,queries_id,qrels_id)


@app.get('/api/real-data/pool')
def candidate_pool(dataset_id:str,queries_id:str):
    values=[p for p in data.listing('candidate_pools') if p['dataset_id']==dataset_id and p['queries_id']==queries_id]
    return values[0] if values else {'records':[]}


@app.get('/api/real-data/source/{dataset_id}')
def source_report(dataset_id:str):
    return data.get('source_reports',dataset_id)


@app.get('/api/real-data/preparation')
def preparation_report():
    try:
        report=data.get('preparation_reports','danang_bm25')
    except FileNotFoundError:
        return {}
    latest=next((d for d in data.listing('datasets') if d['kind']=='qrels' and d['dataset_id']==report['imported']['dataset_id'] and d['queries_id']==report['queries']['queries_id']),None)
    if latest:
        report['selected_qrels_id']=latest['id']
    return report


@app.get('/api/real-data/query-evidence/{queries_id}')
def query_evidence(queries_id:str):
    return data.get('query_evidence',queries_id)


@app.get('/api/templates/{kind}')
def template(kind: str, format: str = 'csv'):
    if kind not in data.SCHEMAS or format not in ('csv', 'xlsx', 'json'):
        raise ValueError('Template không hợp lệ')
    frame = pd.DataFrame(columns=data.SCHEMAS[kind])
    if format == 'json':
        # Empty array plus schema in OpenAPI/README; CSV/XLSX include column headers.
        content, mime = b'[]', 'application/json'
    elif format == 'csv':
        content, mime = frame.to_csv(index=False).encode('utf-8-sig'), 'text/csv'
    else:
        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False)
        content, mime = buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    return Response(content, media_type=mime, headers={'Content-Disposition': f'attachment; filename="{kind}.{format}"'})


@app.get('/api/indexes')
def indexes():
    return data.listing('indexes')


@app.post('/api/indexes')
def index(config: retrieval.IndexConfig):
    ds = data.get('datasets', config.dataset_id)
    if ds['kind'] != 'places':
        raise ValueError('Cần chọn dataset địa điểm')
    return jobs.submit('indexing', lambda job: retrieval.build_index(config, job))


@app.post('/api/models/load')
def load(config: retrieval.IndexConfig):
    if config.method == 'BM25':
        raise ValueError('BM25 không có embedding model')
    return jobs.submit('load_model', lambda job: {'loading_ms': retrieval.manager.load(config), 'model': retrieval.manager.info})


@app.post('/api/models/unload')
def unload():
    return jobs.submit('unload_model', lambda job: retrieval.manager.unload())


class Search(BaseModel):
    index_ids: list[str] = Field(min_length=1)
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=1000)
    metric: str | None = 'Cosine'
    exact: bool = True

    @field_validator('query')
    @classmethod
    def nonblank_query(cls,value):
        value=value.strip()
        if not value:
            raise ValueError('Hãy nhập nội dung truy vấn, không chỉ khoảng trắng')
        return value


@app.post('/api/search')
def search(body: Search):
    indexes = [data.get('indexes', i) for i in body.index_ids]
    first = indexes[0]['config']
    if any(i['status']!='ready' for i in indexes):
        raise ValueError('Phương án đã chọn chưa có index sẵn sàng')
    if any(i['config']['method']!='BM25' for i in indexes):
        body.metric=retrieval.normalize_metric(body.metric)
    if any(i['config']['dataset_id'] != first['dataset_id'] or i['config']['fields'] != first['fields'] for i in indexes):
        raise ValueError('So sánh yêu cầu cùng dataset và các trường indexing')
    return jobs.submit('search', lambda job: [retrieval.search(i, body.query, body.k, body.metric, body.exact) for i in body.index_ids])


class QualitativeNote(BaseModel):
    index_id: str
    place_id: str
    verdict: str = 'unreviewed'
    reason: str = Field(default='',max_length=10000)


class QualitativeReview(BaseModel):
    observation: str = Field(default='',max_length=20000)
    notes: list[QualitativeNote] = Field(default_factory=list)


def qualitative_job(job_id):
    saved=data.get('jobs',job_id)
    if saved['kind']!='search' or saved['status']!='completed' or not isinstance(saved.get('result'),list):
        raise ValueError('Chỉ nhận xét phiên tìm kiếm văn bản đã hoàn tất')
    return saved


@app.get('/api/qualitative/history')
def qualitative_history():
    reviewed={r['id'] for r in data.listing('qualitative_reviews')}
    values=[]
    for job in data.listing('jobs'):
        if job['kind']=='search' and job['status']=='completed' and isinstance(job.get('result'),list):
            answers=job['result']
            values.append(dict(id=job['id'],created_at=job['created_at'],query=answers[0]['query'] if answers else '',
                               methods=[a['method'] for a in answers],reviewed=job['id'] in reviewed))
    return values[:50]


@app.get('/api/qualitative/{job_id}/notes')
def qualitative_notes(job_id:str):
    qualitative_job(job_id)
    try:
        return data.get('qualitative_reviews',job_id)
    except FileNotFoundError:
        return dict(id=job_id,observation='',notes=[])


@app.post('/api/qualitative/{job_id}/notes')
def save_qualitative_notes(job_id:str,body:QualitativeReview):
    saved=qualitative_job(job_id)
    allowed={(a['index_id'],r['place_id']) for a in saved['result'] for r in a['results']}
    keys=set()
    for note in body.notes:
        key=(note.index_id,note.place_id)
        if key not in allowed or key in keys or note.verdict not in ('unreviewed','relevant','partial','irrelevant'):
            raise ValueError('Nhận xét trùng, verdict không hợp lệ hoặc địa điểm không nằm trong kết quả')
        keys.add(key)
    value=dict(id=job_id,created_at=data.now(),query=saved['result'][0]['query'] if saved['result'] else '',
               observation=body.observation,notes=[n.model_dump() for n in body.notes],purpose='qualitative_only')
    data.save('qualitative_reviews',job_id,value)
    return value


@app.post('/api/benchmarks')
def benchmark(body: runner.BenchmarkConfig):
    runner.validate_config(body)
    return jobs.submit('benchmark', lambda job: runner.run(body, job))


@app.post('/api/evaluation/query')
def evaluate_saved_query(body:query_evaluation.Config):
    return query_evaluation.calculate(body)


@app.get('/api/jobs')
def list_jobs():
    return [{k: v for k, v in job.items() if k != 'result'} for job in data.listing('jobs')]


@app.get('/api/jobs/{key}')
def job(key: str):
    return data.ram_only(data.get('jobs', key))


@app.post('/api/jobs/{key}/cancel')
def cancel(key: str):
    active = jobs.ACTIVE.get(key)
    if not active:
        raise ValueError('Tác vụ đã kết thúc hoặc không tồn tại')
    active.cancel.set()
    return {'status': 'cancelling', 'note': 'Dừng tại ranh giới batch/query; thao tác đang chạy phải hoàn tất trước.'}


@app.get('/api/runs')
def runs():
    return data.ram_only([{k: v for k, v in r.items() if k not in ('samples', 'detail', 'cold_start')} for r in data.listing('runs')])


@app.get('/api/runs/{key}')
def run(key: str):
    return data.ram_only(data.get('runs', key))


@app.get('/api/runs/{key}/export')
def export(key: str, format: str = 'json'):
    value = data.ram_only(data.get('runs', key))
    summary = pd.DataFrame(value['summary'])
    detail = pd.DataFrame(value['detail'])
    if format == 'json':
        content, mime = json.dumps(value, ensure_ascii=False, indent=2).encode(), 'application/json'
    elif format == 'csv':
        # A zip preserves provenance and all raw results alongside CSV tables.
        import zipfile
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, frame in [('summary', summary), ('detail', detail)]:
                archive.writestr(name + '.csv', frame.to_csv(index=False).encode('utf-8-sig'))
            archive.writestr('provenance-and-results.json', json.dumps(value, ensure_ascii=False, indent=2))
        content, mime, format = buffer.getvalue(), 'application/zip', 'zip'
    elif format == 'xlsx':
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            summary.to_excel(writer, sheet_name='Summary', index=False)
            detail.to_excel(writer, sheet_name='Per query', index=False)
            pd.DataFrame(value.get('ground_truth',[])).to_excel(writer,sheet_name='Ground Truth',index=False)
            pd.DataFrame(value.get('query_set',[])).to_excel(writer,sheet_name='Queries',index=False)
            try:
                source=data.get('source_reports',value['dataset_id'])
                pd.DataFrame([dict(place_id=pid,filename=source['filename'],source_sha256=source['source_sha256'],**link)
                              for pid,links in source['source_links'].items() for link in links]).to_excel(writer,sheet_name='Source mapping',index=False)
            except FileNotFoundError:
                pass
            raw = [dict(query_id=s['query_id'], repeat=s['repeat'], index_id=s['index_id'], **r) for s in value['samples'] for r in s['results']]
            pd.DataFrame([{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in raw]).to_excel(writer, sheet_name='Rankings', index=False)
            meta = {k: v for k, v in value.items() if k not in ('detail', 'samples','ground_truth','query_set')}
            chunks=[]
            for key,item in meta.items():
                encoded=json.dumps(item,ensure_ascii=False)
                chunks.extend({'key':key,'part':i//30000+1,'value':encoded[i:i+30000]} for i in range(0,len(encoded),30000))
            pd.DataFrame(chunks).to_excel(writer, sheet_name='Provenance', index=False)
        content, mime = buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif format == 'png':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        if not summary.empty:
            names = [r['method'] + ' · ' + r['index_id'][-5:] for r in value['summary']]
            valid = [(n, r['ndcg']) for n, r in zip(names, value['summary']) if r['ndcg'] is not None]
            axes[0].bar([n for n, _ in valid], [v for _, v in valid], color='#0f766e')
            axes[1].bar(names, summary['p95_ms'], color='#6366f1')
        axes[0].set_title(f"nDCG@{value['config']['k']}")
        axes[0].set_ylim(0, 1)
        axes[1].set_title('P95 latency (ms) · warm')
        fig.suptitle(f"Run {key[:8]} · dataset {value['dataset_id'][:8]} · {value['status']}")
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=160)
        plt.close(fig)
        content, mime = buffer.getvalue(), 'image/png'
    else:
        raise ValueError('Định dạng export không hợp lệ')
    return Response(content, media_type=mime, headers={'Content-Disposition': f'attachment; filename="benchmark-{key}.{format}"'})


@app.get('/image-search', include_in_schema=False)
def image_playground():
    return FileResponse(Path(__file__).resolve().parent / 'images' / 'playground.html')


DIST = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='dashboard')
