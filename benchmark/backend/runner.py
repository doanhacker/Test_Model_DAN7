import threading
import time
import uuid
import numpy as np
from pydantic import BaseModel, Field

from . import data
from .metrics import evaluate_query, latency_summary
from .retrieval import search, resources, environment, manager


class BenchmarkConfig(BaseModel):
    index_ids: list[str] = Field(min_length=1)
    queries_id: str
    qrels_id: str
    k: int = Field(default=10, ge=1, le=1000)
    threshold: int = Field(default=1, ge=1, le=3)
    repeats: int = Field(default=3, ge=1, le=100)
    warmup: int = Field(default=1, ge=0, le=20)
    metric: str = 'Cosine'
    exact: bool = True
    official: bool = True
    split: str = 'test'


def validate_config(config):
    if config.metric not in ('Cosine', 'Dot', 'Euclid') or config.split not in ('train', 'validation', 'test', 'all'):
        raise ValueError('Metric/split không hợp lệ')
    if len(config.index_ids) != len(set(config.index_ids)):
        raise ValueError('Index bị chọn trùng')
    indexes = [data.get('indexes', key) for key in config.index_ids]
    first = indexes[0]['config']
    if any(i['status'] != 'ready' or i['config']['dataset_id'] != first['dataset_id'] or i['config']['fields'] != first['fields'] for i in indexes):
        raise ValueError('So sánh yêu cầu index sẵn sàng, cùng dataset version và thứ tự các trường')
    qs = data.get('datasets', config.queries_id)
    gt = data.get('datasets', config.qrels_id)
    if qs['kind'] != 'queries' or gt['kind'] != 'qrels' or gt['dataset_id'] != first['dataset_id'] or gt['queries_id'] != qs['id']:
        raise ValueError('Query/Ground Truth không khớp dataset version')
    selected = [q for q in qs['records'] if config.split == 'all' or q['split'] == config.split]
    if not selected:
        raise ValueError('Không có truy vấn trong split đã chọn')
    ds = data.get('datasets', first['dataset_id'])
    labels = {}
    for r in gt['records']:
        if r.get('review_status','confirmed')=='confirmed' and r['relevance'] is not None:
            labels.setdefault(r['query_id'], {})[r['place_id']] = r['relevance']
    corpus = {p['place_id'] for p in ds['records']}
    if config.official:
        missing = [q['query_id'] for q in selected if set(labels.get(q['query_id'], {})) != corpus]
        if missing:
            missing_pairs=sum(len(corpus)-len(labels.get(q['query_id'],{})) for q in selected)
            raise ValueError(f'Chưa đủ nhãn đã xác nhận: thiếu {missing_pairs:,} cặp query/place trên {len(missing)} truy vấn. Top20 BM25 không đảm bảo đã tìm đủ nhãn dương cho Recall. Bổ sung nhãn hoặc dùng chế độ thăm dò.')
        if not config.exact:
            raise ValueError('Benchmark chính thức dùng exact search để tách sai số ANN')
    return indexes, ds, qs, gt, selected, labels, corpus


def run(config, job):
    indexes, ds, qs, gt, queries, labels, corpus = validate_config(config)
    key = uuid.uuid4().hex
    result = dict(id=key, created_at=data.now(), config=config.model_dump(), dataset_id=ds['id'],
                  dataset_sha256=ds['sha256'], queries_sha256=qs['sha256'], qrels_sha256=gt['sha256'],
                  indexes=indexes, environment=environment(), samples=[], detail=[], summary=[], status='running')
    result['ground_truth'] = gt['records']
    result['query_set'] = qs['records']
    result['label_coverage']=dict(confirmed=sum(len(labels.get(q['query_id'],{})) for q in queries),
                                 missing_corpus=sum(len(corpus)-len(labels.get(q['query_id'],{})) for q in queries),
                                 warning='Chỉ dùng nhãn confirmed. Recall/nDCG thăm dò dựa trên nhãn hiện có; pool BM25 có thiên lệch và không thay thế nhãn toàn corpus.')
    total = len(indexes) * len(queries) * config.repeats
    completed = 0
    peak = resources()
    stop = threading.Event()

    def sample():
        while not stop.wait(.1):
            current = resources()
            for field in ('ram_mb',):
                if current[field] is not None:
                    peak[field] = max(peak[field] or 0, current[field])

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    data.save('runs', key, result)
    started = time.perf_counter()
    try:
        for index in indexes:
            # Explicit unload records the first query as a model cold start.
            if index['config']['method'] != 'BM25':
                manager.unload()
            cold = search(index['id'], queries[0]['query_text'], config.k, config.metric, config.exact)
            result.setdefault('cold_start', []).append(cold)
            for _ in range(config.warmup):
                job.check()
                search(index['id'], queries[0]['query_text'], config.k, config.metric, config.exact)
            for repeat in range(config.repeats):
                for query in queries:
                    job.check()
                    answer = search(index['id'], query['query_text'], config.k, config.metric, config.exact)
                    metric = evaluate_query([r['place_id'] for r in answer['results']], labels.get(query['query_id'], {}), corpus, config.k, config.threshold, config.official)
                    record = dict(index_id=index['id'], method=index['config']['method'], repeat=repeat + 1,
                                  query_id=query['query_id'], query_type=query['query_type'], split=query['split'], **metric, **answer['timing'], **answer['resources'])
                    result['detail'].append(record)
                    result['samples'].append(dict(query_id=query['query_id'], repeat=repeat + 1, **answer))
                    completed += 1
                    job.update(completed, total, f"{record['method']} · {query['query_id']} · lần {repeat + 1}")
            selected = [r for r in result['detail'] if r['index_id'] == index['id']]
            summary = dict(index_id=index['id'], method=index['config']['method'], k=config.k, queries=len(queries),
                           samples=len(selected), eligible_queries=len({r['query_id'] for r in selected if r['ndcg'] is not None}),
                           **latency_summary([r['total_ms'] for r in selected]),
                           ram_mb=max(r['ram_mb'] for r in selected),
                           indexing_ms=index['indexing_ms'], index_size_bytes=index.get('index_size_bytes'),
                           model_loading_ms=cold['timing']['model_loading_ms'],
                           embedding_ms=float(np.mean([r['embedding_ms'] for r in selected])), search_ms=float(np.mean([r['search_ms'] for r in selected])))
            for field in ('precision', 'recall', 'mrr', 'ndcg'):
                values = [r[field] for r in selected if r[field] is not None]
                summary[field] = float(np.mean(values)) if values else None
                summary[field + '_samples'] = len(values)
            result['summary'].append(summary)
            data.save('runs', key, result)
        job.check()
        result['status'] = 'completed'
        return {'run_id': key}
    except BaseException as exc:
        result.update(status='cancelled' if job.cancel.is_set() else 'failed', error=str(exc))
        raise
    finally:
        stop.set()
        monitor.join(timeout=1)
        result.update(duration_ms=(time.perf_counter() - started) * 1000, peak_resources=peak, finished_at=data.now())
        data.save('runs', key, result)
