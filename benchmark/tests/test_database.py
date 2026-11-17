import json
import psycopg
import tempfile
import unittest
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import data,database,retrieval,runner
from backend.main import app


class DatabaseTests(unittest.TestCase):
    def test_rerun_keeps_saved_configuration_and_adds_new_run(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[dict(place_id='p',name='Rerun unit test')],'test')
            qs=data.import_records('queries',[dict(query_id='q',query_text='unit test',query_type='unit',split='test')],'test')
            gt=data.import_records('qrels',[dict(query_id='q',place_id='p',relevance=3,review_status='confirmed')],'test',ds['id'],qs['id'])
            cfg=runner.BenchmarkConfig(index_ids=['idx'],queries_id=qs['id'],qrels_id=gt['id'],repeats=1,warmup=0)
            data.save('indexes','idx',dict(id='idx',config=retrieval.IndexConfig(dataset_id=ds['id']).model_dump(),status='ready',indexing_ms=1))
            old=dict(id='old',created_at='2026-01-01',dataset_id=ds['id'],status='completed',config=cfg.model_dump(),summary=[],detail=[],samples=[])
            data.save('runs','old',old)
            def fake_search(index_id,query,k,metric,exact):
                return dict(index_id=index_id,method='BM25',query=query,results=[dict(place_id='p',name='Unit test',rank=1,score=1.)],
                            timing=dict(total_ms=1,model_loading_ms=0,embedding_ms=0,search_ms=1,phase='warm'),resources=dict(ram_mb=20))
            with TestClient(app) as client,patch('backend.runner.search',side_effect=fake_search):
                response=client.post('/api/database/runs/old/rerun')
                self.assertEqual(response.status_code,200,response.text)
                deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    job=client.get('/api/jobs/'+response.json()['id']).json()
                    if job['status'] not in ('queued','running'):
                        break
                    time.sleep(.05)
                self.assertEqual(job['status'],'completed',job)
                new=data.get('runs',job['result']['run_id'])
                self.assertNotEqual(new['id'],'old')
                self.assertEqual(new['config'],old['config'])
                self.assertEqual(data.get('runs','old'),old)
                self.assertEqual(data.database_info()['counts']['benchmark_runs'],2)

    def test_legacy_migration_is_lossless_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            folder=root/'datasets'
            folder.mkdir()
            value=dict(id='v1',kind='places',name='Migration test',created_at='2026-01-01',records=[dict(place_id='p1',name='Original name',tags=[],amenities=[])])
            source=folder/'v1.json'
            source.write_text(json.dumps(value),encoding='utf-8')
            before=source.read_bytes()
            database.initialize(root)
            database.initialize(root)
            self.assertEqual(database.get(root,'datasets','v1'),value)
            self.assertEqual(database.info(root)['counts']['places'],1)
            self.assertEqual(source.read_bytes(),before)
            # SQLite is authoritative: stale JSON is not reimported on restart.
            database.save(root,'jobs','j1',dict(id='j1',status='completed'))
            self.assertEqual(database.get(root,'jobs','j1')['status'],'completed')

    def test_failed_projection_rolls_back_document_and_relations(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[dict(place_id='p',name='Transaction test')],'test')
            damaged={**ds,'records':[dict(place_id='p',name='A'),dict(place_id='p',name='B')]}
            with self.assertRaises(psycopg.IntegrityError):
                data.save('datasets',ds['id'],damaged)
            self.assertEqual(data.get('datasets',ds['id']),ds)
            self.assertEqual(data.database_info()['counts']['places'],1)

    def test_run_history_nullable_metrics_paging_and_consistent_backup(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[dict(place_id='p',name='Database history test')],'test')
            for number in (1,2):
                key='run'+str(number)
                run=dict(id=key,dataset_id=ds['id'],created_at=f'2026-01-0{number}',status='completed',config=dict(k=10),environment={},
                         summary=[dict(index_id='idx',method='BM25',queries=1,samples=1,recall=None,mrr=None,ndcg=None,average_ms=number*10,ram_mb=100)],
                         detail=[dict(index_id='idx',query_id='q',repeat=1,total_ms=number*10,ram_mb=100)],
                         samples=[dict(index_id='idx',query_id='q',repeat=1,results=[dict(place_id='p',rank=1,score=2.)])])
                data.save('runs',key,run)
                data.save('runs',key,run)
            with TestClient(app) as client:
                info=client.get('/api/database/info').json()
                self.assertEqual(info['counts']['benchmark_runs'],2)
                self.assertEqual(info['counts']['run_metrics'],2)
                self.assertEqual(info['counts']['query_metrics'],2)
                self.assertEqual(info['counts']['search_results'],2)
                history=client.get('/api/database/history',params=dict(method='BM25',page_size=1)).json()
                self.assertEqual(history['total'],2)
                self.assertEqual(history['records'][0]['run_id'],'run2')
                self.assertIsNone(history['records'][0]['recall_at_k'])
                places=client.get('/api/database/places',params=dict(dataset_id=ds['id'],search='history')).json()
                self.assertEqual(places['total'],1)
                self.assertEqual(places['records'][0]['place_id'],'p')
                response=client.get('/api/database/backup')
                self.assertEqual(response.status_code,200)
                text=response.content.decode('utf-8')
                self.assertIn('PostgreSQL benchmark backup',text)
                self.assertIn('COPY "places"',text)
                self.assertIn('COPY "run_metrics"',text)
                self.assertTrue(text.endswith('COMMIT;\n'))
                self.assertEqual(info['engine'],'PostgreSQL')
