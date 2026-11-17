import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,MagicMock

import pandas as pd
from PIL import Image
from fastapi.testclient import TestClient
from backend import data,database,place_media
from backend.main import app


class PlaceMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=patch.object(data,'ROOT',Path(self.temp.name))
        self.root.start()
        self.ds=data.import_records('places',[dict(place_id='p',name='Unit photo test')],'test only')
        self.frame=pd.DataFrame([{'Mã địa điểm':'p','Thành phố':'Đà Nẵng',
              'Liên kết ảnh đại diện':'https://lh3.googleusercontent.com/unit-cover',
              'Liên kết tất cả ảnh':'https://lh3.googleusercontent.com/unit-cover https://lh3.googleusercontent.com/unit-next'}])

    def tearDown(self):
        self.root.stop()
        self.temp.cleanup()

    def import_photos(self):
        book=MagicMock(sheet_names=['Địa điểm chung'])
        with patch.object(place_media.pd,'ExcelFile',return_value=book),patch.object(place_media.pd,'read_excel',return_value=self.frame):
            return place_media.import_excel(b'unit source','unit.xlsx',self.ds['id'])

    def test_upsert_galleries_and_search_attachment(self):
        result=self.import_photos()
        self.import_photos()
        self.assertEqual(result['image_links'],2)
        self.assertEqual(result['places_with_images'],1)
        self.assertEqual(data.database_info()['counts']['place_images'],2)
        record=place_media.attach(self.ds['id'],[dict(place_id='p',name='Unit')])[0]
        self.assertEqual(record['image_count'],2)
        self.assertTrue(record['images'][0]['is_cover'])
        self.assertTrue(record['image_url'].startswith('/api/place-images/'))
        self.assertIsNone(place_media.attach(self.ds['id'],[dict(place_id='other')])[0]['image_url'])

    def test_photo_bytes_cached_in_postgres_and_reused(self):
        self.import_photos()
        photo=place_media.galleries(self.ds['id'],['p'])['p'][0]
        buffer=io.BytesIO()
        Image.new('RGB',(64,48),'blue').save(buffer,format='PNG')
        with TestClient(app) as client,patch.object(place_media,'download',return_value=buffer.getvalue()) as download:
            first=client.get(photo['url'])
            second=client.get(photo['url'])
            self.assertEqual(first.status_code,200)
            self.assertEqual(first.headers['content-type'],'image/jpeg')
            self.assertEqual(first.content,second.content)
            self.assertEqual(download.call_count,1)
            self.assertEqual(Image.open(io.BytesIO(first.content)).size,(64,48))
            with database.session(data.ROOT,data.db_path_override()) as conn:
                stored=conn.execute('SELECT image_content,content_sha256 FROM place_images WHERE image_id=?',(photo['id'],)).fetchone()
            self.assertEqual(bytes(stored[0]),first.content)
            self.assertEqual(stored[1],hashlib.sha256(first.content).hexdigest())
        self.import_photos()
        self.assertEqual(place_media.stats(self.ds['id'])['cached_images'],1)

    def test_reject_non_source_hosts_and_handle_bad_image(self):
        self.assertFalse(place_media.allowed_url('http://127.0.0.1/private'))
        self.assertFalse(place_media.allowed_url('https://example.com/file'))
        self.assertFalse(place_media.allowed_url('https://lh3.googleusercontent.com:8443/file'))
        self.import_photos()
        photo=place_media.galleries(self.ds['id'],['p'])['p'][0]
        with TestClient(app) as client,patch.object(place_media,'download',return_value=b'not an image'):
            self.assertEqual(client.get(photo['url']).status_code,404)
        self.assertEqual(place_media.stats(self.ds['id'])['failed_images'],1)
