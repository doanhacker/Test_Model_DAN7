"""Evaluate one real saved query search using explicitly confirmed judgments."""
import uuid
from pydantic import BaseModel,Field
from . import data
from .metrics import evaluate_query
from .retrieval import environment


class Config(BaseModel):
    job_id:str
    query_id:str
    qrels_id:str
    k:int=Field(default=10,ge=1,le=1000)
    threshold:int=Field(default=1,ge=1,le=3)
    official:bool=False


def calculate(config):
    job=data.get('jobs',config.job_id)
    if job['kind']!='search' or job['status']!='completed' or not isinstance(job.get('result'),list) or not job['result']:
        raise ValueError('Cần phiên tìm kiếm đã hoàn tất')
    gt=data.get('datasets',config.qrels_id)
    if gt['kind']!='qrels':
        raise ValueError('Cần phiên bản Ground Truth')
    queries=data.get('datasets',gt['queries_id'])
    query=next((q for q in queries['records'] if q['query_id']==config.query_id),None)
    if not query:
        raise ValueError('query_id không thuộc bộ truy vấn của Ground Truth')
    ds=data.get('datasets',gt['dataset_id'])
    corpus={p['place_id'] for p in ds['records']}
    labels={r['place_id']:r['relevance'] for r in gt['records'] if r['query_id']==config.query_id and r.get('review_status','confirmed')=='confirmed' and r['relevance'] is not None}
    details=[]
    summaries=[]
    samples=[]
    indexes=[]
    for answer in job['result']:
        index=data.get('indexes',answer['index_id'])
        if index['config']['dataset_id']!=ds['id'] or answer['query'].strip()!=query['query_text'].strip():
            raise ValueError('Nội dung truy vấn hoặc dataset không khớp phiên tìm kiếm; hãy chạy lại câu đã chọn')
        if config.k>answer.get('k',len(answer['results'])):
            raise ValueError('Top-K đánh giá vượt số K đã tìm kiếm; hãy lấy thêm kết quả trước')
        metrics=evaluate_query([r['place_id'] for r in answer['results']],labels,corpus,config.k,config.threshold,config.official)
        pool_ids={r['place_id'] for r in answer['results']}
        metrics.update(pool_size=len(pool_ids),judged_pool=len(pool_ids&set(labels)),missing_pool=len(pool_ids-set(labels)))
        if not config.official and metrics['missing_pool']:
            # Prevent the trivial "Recall=1" when only the evaluated Top-K has
            # been judged but the larger candidate pool has not been reviewed.
            metrics.update(recall=None,ndcg=None,idcg=None,recall_status='needs_pool_labels',ndcg_status='needs_pool_labels',status='incomplete_labels')
        details.append(dict(index_id=answer['index_id'],method=answer['method'],query_id=query['query_id'],query_type=query['query_type'],split=query['split'],repeat=1,**metrics,**answer['timing'],**answer['resources']))
        summaries.append(dict(index_id=answer['index_id'],method=answer['method'],k=config.k,queries=1,samples=1,
                              eligible_queries=int(metrics['ndcg'] is not None),**{m:metrics[m] for m in ('precision','recall','mrr','ndcg')},
                              **{m+'_samples':int(metrics[m] is not None) for m in ('precision','recall','mrr','ndcg')},
                              recall_status=metrics['recall_status'],mrr_status=metrics['mrr_status'],ndcg_status=metrics['ndcg_status'],
                              average_ms=answer['timing']['total_ms'],p50_ms=answer['timing']['total_ms'],p95_ms=answer['timing']['total_ms'],p99_ms=answer['timing']['total_ms'],ram_mb=answer['resources']['ram_mb']))
        samples.append(dict(query_id=query['query_id'],repeat=1,**answer))
        indexes.append(index)
    key=uuid.uuid4().hex
    result=dict(id=key,created_at=data.now(),finished_at=data.now(),status='completed',dataset_id=ds['id'],
                dataset_sha256=ds['sha256'],queries_sha256=queries['sha256'],qrels_sha256=gt['sha256'],
                config=dict(index_ids=[i['id'] for i in indexes],queries_id=queries['id'],qrels_id=gt['id'],k=config.k,threshold=config.threshold,
                            official=config.official,repeats=1,warmup=0,split=query['split'],source_job_id=job['id'],query_id=query['query_id']),
                evaluation_only=True,measurement_note='Đánh giá thứ hạng đã lưu; timing/RAM là số đo của phiên tìm kiếm nguồn, không đo lại.',
                environment=environment(),indexes=indexes,query_set=[query],ground_truth=gt['records'],samples=samples,detail=details,summary=summaries,
                label_coverage=dict(confirmed=len(labels),missing_corpus=len(corpus)-len(labels),
                                    warning='MRR có thể chính xác khi biết kết quả đúng đầu tiên và các vị trí trước nó. Recall/nDCG theo pool chỉ là thăm dò, không phải kết luận toàn corpus.'))
    data.save('runs',key,result)
    return result
