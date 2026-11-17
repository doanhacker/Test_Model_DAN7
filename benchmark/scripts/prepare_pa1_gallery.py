"""Prepare a real PA1 Gallery through existing APIs, retaining evaluation indexes.

python scripts/prepare_pa1_gallery.py --cached
python scripts/prepare_pa1_gallery.py --all
"""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

import httpx

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(cached,base_url,dataset_id):
    from backend import data,database
    client=httpx.Client(base_url=base_url,timeout=120)
    report={'id':uuid.uuid4().hex,'created_at':data.now(),'dataset_id':dataset_id,'cached_only':cached,
            'status':'running','image_ids':[],'errors':[],'imported':0,'reused':0,'failed':0,'total':0}
    target=ROOT/'runtime'/('pa1-gallery-cached.json' if cached else 'pa1-gallery-all.json')
    def save():
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        data.save('image_imports',report['id'],report)
    def request(path,body=None):
        response=client.get(path) if body is None else client.post(path,json=body)
        response.raise_for_status()
        return response.json()
    def wait(identity):
        report['job_id']=identity
        save()
        previous=None
        while True:
            job=request('/api/jobs/'+identity)
            state=(job['status'],job.get('current'),job.get('total'))
            if state!=previous:
                print(json.dumps({'job':identity,'status':state[0],'current':state[1],'total':state[2]}),flush=True)
                previous=state
            if job['status'] not in ('running','queued'):
                if job['status']!='completed':
                    raise RuntimeError(job.get('error',job['status']))
                return job['result']
            time.sleep(3)
    try:
        if cached:
            with database.session(data.ROOT,data.db_path_override()) as conn:
                rows=conn.execute('''SELECT i.image_id,i.place_id,p.name FROM place_images i JOIN places p
                                    ON p.dataset_id=i.dataset_id AND p.place_id=i.place_id
                                    WHERE i.dataset_id=? AND i.image_content IS NOT NULL ORDER BY p.ordinal,i.ordinal,i.image_id''',(dataset_id,)).fetchall()
            report['source_images']=len(rows)
            for number,row in enumerate(rows,1):
                try:
                    raw=client.get('/api/place-images/'+row['image_id'])
                    raw.raise_for_status()
                    uploaded=client.post('/api/ai/images/assets',files={'file':('source.jpg',raw.content,'image/jpeg')},
                                         data={'place_id':row['place_id'],'dataset_id':dataset_id})
                    uploaded.raise_for_status()
                    asset=uploaded.json()
                    if asset['id'] not in report['image_ids']:
                        report['image_ids'].append(asset['id'])
                    data.save('image_source_links',row['image_id'],dict(id=row['image_id'],image_id=asset['id'],
                              source_image_id=row['image_id'],dataset_id=dataset_id,place_id=row['place_id'],created_at=data.now()))
                except Exception as exc:
                    report['errors'].append({'source_image_id':row['image_id'],'place_id':row['place_id'],'error':str(exc)})
                if number%20==0 or number==len(rows):
                    print(json.dumps({'phase':'cached_import','current':number,'total':len(rows),'gallery_images':len(report['image_ids']),'errors':len(report['errors'])}),flush=True)
                    save()
        else:
            # Download/register through read/upload endpoints outside the shared
            # inference queue, so the user can search the ready cached Gallery.
            with database.session(data.ROOT,data.db_path_override()) as conn:
                rows=conn.execute('''SELECT i.image_id,i.place_id FROM place_images i JOIN places p
                                    ON p.dataset_id=i.dataset_id AND p.place_id=i.place_id
                                    WHERE i.dataset_id=? ORDER BY p.ordinal,i.ordinal,i.image_id''',(dataset_id,)).fetchall()
            report.update(source_images=len(rows),total=len(rows),phase='importing')
            known={a['id'] for a in data.listing('image_assets')}
            save()
            for number,row in enumerate(rows,1):
                try:
                    try:
                        link=data.get('image_source_links',row['image_id'])
                        asset=data.get('image_assets',link['image_id'])
                        report['reused']+=1
                    except FileNotFoundError:
                        raw=client.get('/api/place-images/'+row['image_id']);raw.raise_for_status()
                        uploaded=client.post('/api/ai/images/assets',files={'file':('source.jpg',raw.content,'image/jpeg')},
                                             data={'place_id':row['place_id'],'dataset_id':dataset_id})
                        uploaded.raise_for_status();asset=uploaded.json()
                        report['reused' if asset['id'] in known else 'imported']+=1
                        known.add(asset['id'])
                        data.save('image_source_links',row['image_id'],dict(id=row['image_id'],image_id=asset['id'],
                                  source_image_id=row['image_id'],dataset_id=dataset_id,place_id=row['place_id'],created_at=data.now()))
                    if asset['id'] not in report['image_ids']:
                        report['image_ids'].append(asset['id'])
                except Exception as exc:
                    report['failed']+=1
                    report['errors'].append({'source_image_id':row['image_id'],'place_id':row['place_id'],'error':str(exc)})
                report['current']=number
                if number%10==0 or number==len(rows):
                    save()
                    print(json.dumps({'phase':'full_import','current':number,'total':len(rows),'gallery_images':len(report['image_ids']),'errors':len(report['errors'])}),flush=True)
        if not report['image_ids']:
            raise RuntimeError('No usable source images were imported')
        report['phase']='indexing'
        # Independent namespace: existing 16-image benchmark manifests stay valid.
        namespace='danang-pa1-cached' if cached else 'danang-pa1-all'
        job=request('/api/ai/images/ablation',{'image_ids':report['image_ids'],
                    'config':{'model':'handcrafted','device':'cpu','batch_size':4,'collection':namespace}})
        report['indexes']=wait(job['id'])['indexes']
        report.update(status='completed',gallery_images=len(report['image_ids']))
        print(json.dumps({'status':'completed','gallery_images':report['gallery_images'],'source_errors':len(report['errors']),'indexes':report['indexes']}),flush=True)
    except BaseException as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        save()
        client.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cached',action='store_true')
    mode.add_argument('--all',action='store_true')
    parser.add_argument('--base-url',default='http://127.0.0.1:8000')
    parser.add_argument('--dataset-id',default='982441763d72185252745cdc')
    args=parser.parse_args()
    run(args.cached,args.base_url,args.dataset_id)
