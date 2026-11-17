"""Source-linked place galleries with PostgreSQL image-byte caching on first view."""
import hashlib
import io
import os
import re
import threading
import warnings
from datetime import datetime,timezone
from urllib.parse import urlsplit

import httpx
import pandas as pd
from PIL import Image,ImageOps,UnidentifiedImageError
from fastapi import APIRouter,HTTPException
from fastapi.responses import Response

from . import data,database

DEFAULT_HOSTS=('lh3.googleusercontent.com','streetviewpixels-pa.googleapis.com','photos.hotelbeds.com',
               'images.trvl-media.com','dynamic-media-cdn.tripadvisor.com','cdn.worldota.net',
               'cdn.ostrovok.ru','d2hyz2bfif3cr8.cloudfront.net')
DOWNLOADS=threading.BoundedSemaphore(4)
router=APIRouter(prefix='/api/place-images',tags=['Place photos'])


def allowed_url(url):
    parts=urlsplit(url)
    hosts=set(os.getenv('PLACE_IMAGE_ALLOWED_HOSTS',','.join(DEFAULT_HOSTS)).split(','))
    try:
        return parts.scheme=='https' and parts.hostname in hosts and not parts.username and not parts.password and parts.port in (None,443)
    except ValueError:
        return False


def import_excel(raw,filename,dataset_id,city='Đà Nẵng'):
    book=pd.ExcelFile(io.BytesIO(raw))
    sheet='Địa điểm chung' if 'Địa điểm chung' in book.sheet_names else book.sheet_names[0]
    frame=pd.read_excel(book,sheet_name=sheet,dtype=str,keep_default_na=False)
    if 'Mã địa điểm' not in frame:
        raise ValueError('Thiếu Mã địa điểm để liên kết ảnh')
    with database.session(data.ROOT,data.db_path_override()) as conn:
        valid={r[0] for r in conn.execute('SELECT place_id FROM places WHERE dataset_id=?',(dataset_id,)).fetchall()}
    images={}
    covers={}
    rejected=0
    for number,record in enumerate(frame.to_dict('records'),2):
        pid=record['Mã địa điểm'].strip()
        if pid not in valid or city and record.get('Thành phố','').strip()!=city:
            continue
        cover=record.get('Liên kết ảnh đại diện','').strip()
        urls=([cover] if cover else [])+record.get('Liên kết tất cả ảnh','').split()
        for ordinal,url in enumerate(dict.fromkeys(urls)):
            if not allowed_url(url):
                rejected+=1
                continue
            covers.setdefault(pid,url)
            identity=hashlib.sha256((dataset_id+'\0'+pid+'\0'+url).encode()).hexdigest()[:32]
            images.setdefault(identity,(identity,dataset_id,pid,url,hashlib.sha256(url.encode()).hexdigest(),url==covers[pid],ordinal,filename,sheet,number))
    with database.session(data.ROOT,data.db_path_override()) as conn:
        conn.executemany('''INSERT INTO place_images(image_id,dataset_id,place_id,source_url,url_sha256,is_cover,ordinal,source_filename,source_sheet,source_row)
                            VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(image_id) DO UPDATE SET is_cover=excluded.is_cover,
                            source_filename=excluded.source_filename,source_sheet=excluded.source_sheet,source_row=excluded.source_row''',list(images.values()))
    result=stats(dataset_id)
    result.update(rejected_urls=rejected,source_sha256=hashlib.sha256(raw).hexdigest())
    return result


def stats(dataset_id):
    with database.session(data.ROOT,data.db_path_override()) as conn:
        result=dict(conn.execute('''SELECT COUNT(*) AS image_links,COUNT(DISTINCT place_id) AS places_with_images,
                         COUNT(*) FILTER (WHERE image_content IS NOT NULL) AS cached_images,
                         COUNT(*) FILTER (WHERE status='failed') AS failed_images
                         FROM place_images WHERE dataset_id=?''',(dataset_id,)).fetchone())
        total=conn.execute('SELECT COUNT(*) FROM places WHERE dataset_id=?',(dataset_id,)).fetchone()[0]
    return dict(dataset_id=dataset_id,**result,places_without_images=total-result['places_with_images'])


def galleries(dataset_id,place_ids):
    if not place_ids:
        return {}
    with database.session(data.ROOT,data.db_path_override()) as conn:
        rows=conn.execute('''SELECT image_id,place_id,source_url,is_cover,status FROM place_images
                             WHERE dataset_id=? AND place_id=ANY(?) ORDER BY is_cover DESC,ordinal,image_id''',(dataset_id,list(place_ids))).fetchall()
    result={}
    for row in rows:
        result.setdefault(row['place_id'],[]).append(dict(id=row['image_id'],url='/api/place-images/'+row['image_id'],
                    source_url=row['source_url'],is_cover=row['is_cover'],status=row['status']))
    return result


def attach(dataset_id,records):
    values=galleries(dataset_id,[r['place_id'] for r in records])
    return [dict(r,images=values.get(r['place_id'],[]),image_url=values[r['place_id']][0]['url'] if values.get(r['place_id']) else None,
                 image_count=len(values.get(r['place_id'],[]))) for r in records]


def download(url):
    with httpx.Client(timeout=httpx.Timeout(10,connect=5),follow_redirects=False) as client:
        for _ in range(4):
            if not allowed_url(url):
                raise ValueError('Nguồn ảnh không nằm trong danh sách cho phép')
            with client.stream('GET',url,headers={'User-Agent':'TourismSearchLab/1.0'}) as response:
                if response.is_redirect:
                    url=str(response.url.join(response.headers['location']))
                    continue
                response.raise_for_status()
                content=bytearray()
                for block in response.iter_bytes():
                    content.extend(block)
                    if len(content)>12*1024*1024:
                        raise ValueError('Ảnh vượt 12 MB')
                return bytes(content)
    raise ValueError('Ảnh chuyển hướng quá nhiều lần')


def thumbnail(raw):
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as source:
            if source.width*source.height>25000000:
                raise ValueError('Ảnh quá lớn để hiển thị')
            source.load()
            image=ImageOps.exif_transpose(source).convert('RGBA')
            image=Image.alpha_composite(Image.new('RGBA',image.size,'white'),image).convert('RGB')
            image.thumbnail((960,640),Image.Resampling.LANCZOS)
            output=io.BytesIO()
            image.save(output,format='JPEG',quality=85)
            return output.getvalue()


@router.get('')
def list_images(dataset_id:str,place_id:str):
    images=galleries(dataset_id,[place_id]).get(place_id,[])
    return dict(place_id=place_id,images=images,image_count=len(images),image_url=images[0]['url'] if images else None)


@router.get('/stats')
def image_stats(dataset_id:str):
    return stats(dataset_id)


@router.get('/{image_id}')
def image_content(image_id:str):
    if not re.fullmatch('[0-9a-f]{32}',image_id):
        raise HTTPException(404,'Không tìm thấy ảnh')
    with database.session(data.ROOT,data.db_path_override()) as conn:
        row=conn.execute('SELECT source_url,image_content,mime_type,content_sha256,status,updated_at FROM place_images WHERE image_id=?',(image_id,)).fetchone()
    if row is None:
        raise HTTPException(404,'Không tìm thấy ảnh')
    content=row['image_content']
    if content is None and row['status']=='failed' and row['updated_at']:
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(row['updated_at'])).total_seconds()
        if age<300:
            raise HTTPException(404,'Ảnh nguồn tạm thời không khả dụng')
    if content is None:
        # Recheck after admission; another request may already have cached the image.
        with DOWNLOADS:
            with database.session(data.ROOT,data.db_path_override()) as conn:
                cached=conn.execute('SELECT image_content FROM place_images WHERE image_id=?',(image_id,)).fetchone()
            content=cached[0]
            if content is None:
                try:
                    content=thumbnail(download(row['source_url']))
                    with database.session(data.ROOT,data.db_path_override()) as conn:
                        conn.execute("UPDATE place_images SET image_content=?,mime_type='image/jpeg',content_sha256=?,status='cached',last_error=NULL,updated_at=? WHERE image_id=?",
                                     (content,hashlib.sha256(content).hexdigest(),data.now(),image_id))
                except (httpx.HTTPError,ValueError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:
                    with database.session(data.ROOT,data.db_path_override()) as conn:
                        conn.execute("UPDATE place_images SET status='failed',last_error=?,updated_at=? WHERE image_id=?",(str(exc)[:1000],data.now(),image_id))
                    raise HTTPException(404,'Ảnh nguồn hiện không tải được') from exc
    return Response(bytes(content),media_type='image/jpeg',headers={'Cache-Control':'private, max-age=86400',
                     'ETag':'"'+hashlib.sha256(content).hexdigest()+'"'})
