"""Restartable, source-linked image import into the existing asset store."""
import uuid

from .. import data,database,place_media
from . import store


def import_photos(dataset_id,place_ids,limit,all_images,job):
    dataset=data.get('datasets',dataset_id)
    if dataset['kind']!='places':
        raise ValueError('Cần dataset địa điểm')
    valid={p['place_id'] for p in dataset['records']}
    if place_ids is not None and (not place_ids or set(place_ids)-valid):
        raise ValueError('Danh sách place_id rỗng hoặc không thuộc dataset')
    where='i.dataset_id=?'
    args=[dataset_id]
    if place_ids is not None:
        where+=' AND i.place_id=ANY(?)'
        args.append(place_ids)
    if not all_images:
        where+=' AND i.is_cover=TRUE'
    with database.session(data.ROOT,data.db_path_override()) as conn:
        rows=conn.execute('SELECT i.image_id,i.place_id,p.name FROM place_images i JOIN places p ON p.dataset_id=i.dataset_id AND p.place_id=i.place_id WHERE '+where+' ORDER BY p.ordinal,i.ordinal,i.image_id LIMIT ?',args+[limit]).fetchall()
    report=dict(id=uuid.uuid4().hex,created_at=data.now(),dataset_id=dataset_id,status='running',
                total=len(rows),imported=0,reused=0,failed=0,image_ids=[],errors=[])
    data.save('image_imports',report['id'],report)
    known_assets={a['id'] for a in data.listing('image_assets')}
    try:
        for number,row in enumerate(rows,1):
            job.check()
            try:
                try:
                    link=data.get('image_source_links',row['image_id'])
                    asset=data.get('image_assets',link['image_id'])
                    report['reused']+=1
                except FileNotFoundError:
                    content=place_media.image_content(row['image_id']).body
                    asset=store.register(content,row['place_id'],row['name'],dataset_id)
                    data.save('image_source_links',row['image_id'],dict(id=row['image_id'],image_id=asset['id'],
                              source_image_id=row['image_id'],dataset_id=dataset_id,place_id=row['place_id'],created_at=data.now()))
                    report['reused' if asset['id'] in known_assets else 'imported']+=1
                    known_assets.add(asset['id'])
                if asset['id'] not in report['image_ids']:
                    report['image_ids'].append(asset['id'])
            except Exception as exc:
                report['failed']+=1
                report['errors'].append(dict(source_image_id=row['image_id'],place_id=row['place_id'],name=row['name'],error=str(exc)))
            job.update(number,len(rows),f"Import ảnh nguồn · {number}/{len(rows)} · lỗi {report['failed']}")
            if number%10==0:
                data.save('image_imports',report['id'],report)
        report['status']='completed'
        return report
    except BaseException as exc:
        report.update(status='cancelled' if job.cancel.is_set() else 'failed',error=str(exc))
        raise
    finally:
        report['finished_at']=data.now()
        data.save('image_imports',report['id'],report)
