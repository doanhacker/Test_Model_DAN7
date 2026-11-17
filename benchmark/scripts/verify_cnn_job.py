"""Verify three real pretrained CNN indexes, upload search, and timing trial."""
import json
import math
from pathlib import Path
import sys
import time

import httpx

ROOT=Path(__file__).resolve().parents[1]


def main(identity):
    c=httpx.Client(base_url='http://127.0.0.1:8000',timeout=120)
    report=dict(job_id=identity,status='running',checks=[])
    target=ROOT/'runtime'/'cnn-real-validation.json'
    def get(path):
        r=c.get(path);r.raise_for_status();return r.json()
    def post(path,body):
        r=c.post(path,json=body);r.raise_for_status();return r.json()
    def wait(job_id):
        while True:
            job=get('/api/jobs/'+job_id)
            if job['status'] not in ('queued','running'):
                if job['status']!='completed':raise RuntimeError(job.get('error',job['status']))
                return job['result']
            time.sleep(4)
    try:
        prepared=wait(identity)
        ids=[prepared['baseline_index_id']]+[i['index_id'] for i in prepared['indexes']]
        indexes=[i for i in get('/api/ai/images/indexes') if i['id'] in ids]
        if len(indexes)!=4 or len({i['gallery_sha256'] for i in indexes})!=1:
            raise RuntimeError('CNN and baseline Gallery snapshots differ')
        gallery=list(indexes[0]['images'])
        source=json.loads((ROOT/'runtime'/'pa1-real-validation.json').read_text(encoding='utf-8'))
        query=source['query_id']
        raw=c.get('/api/ai/images/assets/'+query+'/content');raw.raise_for_status()
        expected={'mobilenet_v3_large':960,'resnet50':2048,'efficientnet_b0':1280}
        for index in indexes:
            if index['status']!='ready' or index['indexed_count']!=len(gallery):raise RuntimeError('Incomplete index')
            model=index['config']['model']
            f=get('/api/ai/images/indexes/'+index['id']+'/vectors/'+gallery[0])
            if model in expected:
                if f['dimension']!=expected[model] or not index['model'].get('frozen'):
                    raise RuntimeError('CNN dimension or pretrained freeze metadata incorrect')
            if not math.isclose(f['statistics']['l2_norm'],1,abs_tol=1e-5):raise RuntimeError('Vector is not normalized')
            response=c.post('/api/ai/images/search',files={'file':('real-query.png',raw.content,'image/png')},data={'index_id':index['id'],'top_k':10})
            response.raise_for_status();answer=wait(response.json()['id'])
            if len(answer['results'])!=10 or not all(math.isfinite(r['score']) for r in answer['results']):raise RuntimeError('Search failed')
            report['checks'].append(dict(model=model,index_id=index['id'],dimension=f['dimension'],count=len(index['images']),
                                         norm=f['statistics']['l2_norm'],search=answer))
        dataset=post('/api/ai/images/datasets',dict(name='PA2 CNN - real Gallery - PENDING LABELS',gallery_image_ids=gallery,
                     queries=[dict(query_id='cnn-source-query-01',image_id=query,exact_place_ids=[])],visual_target='place',judgments=[]))
        cfg=dict(dataset_id=dataset['id'],index_ids=ids,objective='visual',official=True,repeats=3,warmup=1,threshold=1)
        r=c.post('/api/ai/images/benchmarks',json=cfg)
        if r.status_code!=422:raise RuntimeError('Unlabeled official evaluation was not blocked')
        cfg['official']=False
        trial=wait(post('/api/ai/images/benchmarks',cfg)['id'])
        run=get('/api/ai/images/runs/'+trial['image_run_id'])
        if any(s.get(k) is not None for s in run['summary'] for k in ('recall@10','mrr@10','ndcg@10')):
            raise RuntimeError('Unlabeled trial has quality scores')
        report.update(status='passed',gallery_images=len(gallery),dataset_id=dataset['id'],run_id=run['id'],
                      samples=len(run['detail']),trial_summary=run['summary'],confirmed_labels=0,
                      note='Real pretrained inference, vectors and measured performance; quality awaits human judgments.')
    except BaseException as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');c.close()


if __name__=='__main__':
    main(sys.argv[1])
