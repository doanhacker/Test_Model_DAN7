"""Run with python -m backend.prepare_real_bm25; uses only the supplied Excel."""
import json
import threading
from pathlib import Path

from . import data, real_data, retrieval, runner


class Progress:
    cancel=threading.Event()
    def check(self):
        if self.cancel.is_set():
            raise RuntimeError('Cancelled')
    def update(self,current,total,message):
        if current%10==0 or current==total:
            print(f'{current}/{total}: {message}',flush=True)


def main():
    path=Path(__file__).resolve().parents[2]/'data'/'danang_merged_serpapi_da_lam_sach.xlsx'
    before=path.read_bytes()
    imported=real_data.import_excel(before,path.name)
    print(json.dumps(imported,ensure_ascii=False),flush=True)
    cfg=retrieval.IndexConfig(dataset_id=imported['dataset_id'],method='BM25',fields=real_data.SEARCH_FIELDS)
    existing=next((i for i in data.listing('indexes') if i['status']=='ready' and i['config']==cfg.model_dump()),None)
    index=existing or retrieval.build_index(cfg,Progress())
    count=retrieval.es('GET',index['id']+'/_count')['count']
    if count!=imported['valid_places']:
        raise ValueError('Indexed count differs from imported dataset')
    queries=real_data.generate_queries(imported['dataset_id'])
    pool=real_data.create_pool(index['id'],queries['queries_id'],Progress())
    config=runner.BenchmarkConfig(index_ids=[index['id']],queries_id=queries['queries_id'],qrels_id=pool['qrels_id'],
                                  k=10,repeats=1,warmup=1,official=False,split='all')
    run=runner.run(config,Progress())
    coverage=real_data.coverage(imported['dataset_id'],queries['queries_id'],pool['qrels_id'])
    report=dict(imported=imported,index_id=index['id'],indexed_count=count,queries=queries,pool=pool,run=run,coverage=coverage,
                note='Latency/RAM đo thực. Metric chất lượng để null cho tới khi có nhãn được người dùng xác nhận.')
    if path.read_bytes()!=before:
        raise RuntimeError('Source workbook changed during preparation')
    data.save('preparation_reports','danang_bm25',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
