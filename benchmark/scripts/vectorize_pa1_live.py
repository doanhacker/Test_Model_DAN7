"""Incrementally vectorize imported real Gallery photos without long queue locks."""
import argparse
import json
from pathlib import Path
import sys
import time

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(watch):
    from backend import data
    cfg=dict(model='handcrafted',device='cpu',batch_size=4,collection='danang-pa1-live')
    dataset='982441763d72185252745cdc'
    report=dict(id='pa1-live',dataset_id=dataset,created_at=data.now(),status='running',indexed=0)
    target=ROOT/'runtime'/'pa1-live-vectorization.json'
    def save():
        report['updated_at']=data.now()
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        data.save('image_vectorizations',report['id'],report)
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=120) as c:
        def get(path):
            r=c.get(path);r.raise_for_status();return r.json()
        try:
            while True:
                assets=[a for a in get('/api/ai/images/assets') if a.get('dataset_id')==dataset and a.get('place_id')]
                indexes=get('/api/ai/images/indexes')
                meta=next((i for i in indexes if i.get('namespace')==cfg['collection'] and i['config']['model']=='handcrafted'),None)
                done=set(meta['images']) if meta else set()
                pending=[a['id'] for a in assets if a['id'] not in done]
                report.update(indexed=len(done),available=len(assets),remaining=len(pending),index_id=meta['id'] if meta else None)
                save()
                if not pending and (not meta or meta['status']=='ready'):
                    source_file=ROOT/'runtime'/'pa1-gallery-all.json'
                    source=json.loads(source_file.read_text(encoding='utf-8')) if source_file.exists() else {}
                    if not watch or source.get('status')!='running':
                        report['status']='completed';save();return
                    time.sleep(15)
                    continue
                batch=pending[:64] or [assets[0]['id']]
                r=c.post('/api/ai/images/index-assets',json={'image_ids':batch,'config':cfg});r.raise_for_status()
                identity=r.json()['id'];report['job_id']=identity;save()
                while True:
                    j=get('/api/jobs/'+identity)
                    if j['status'] not in ('queued','running'):
                        if j['status']!='completed':raise RuntimeError(j.get('error',j['status']))
                        result=j['result'];report.update(indexed=result['total'],index_id=result['index_id'])
                        save();print(json.dumps({'indexed':result['total'],'index_id':result['index_id']}),flush=True)
                        break
                    time.sleep(2)
                # Give interactive searches an opportunity to enter the queue.
                time.sleep(3)
        except BaseException as exc:
            report.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    run(parser.parse_args().watch)
