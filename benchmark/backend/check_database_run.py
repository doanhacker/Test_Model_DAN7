"""Run the real BM25 query set once, storing a separate PostgreSQL run."""
import json
from . import data,runner
from .prepare_real_bm25 import Progress


def main():
    report=data.get('preparation_reports','danang_bm25')
    before={r['id'] for r in data.listing('runs')}
    config=runner.BenchmarkConfig(index_ids=[report['index_id']],queries_id=report['queries']['queries_id'],
                                 qrels_id=report['pool']['qrels_id'],k=10,repeats=1,warmup=1,official=False,split='all')
    result=runner.run(config,Progress())
    run=data.get('runs',result['run_id'])
    assert result['run_id'] not in before
    assert len(run['detail'])==100
    report['run']=result
    data.save('preparation_reports','danang_bm25',report)
    print(json.dumps(dict(new_run=result,summary=run['summary'],database=data.database_info()),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
