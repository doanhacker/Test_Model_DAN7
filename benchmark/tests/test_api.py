import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient
from backend import data, retrieval, runner
from backend.main import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = data.ROOT
        data.ROOT = Path(self.folder.name)
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        data.ROOT = self.root
        self.folder.cleanup()

    def import_json(self, kind, rows, **extra):
        return self.client.post('/api/import', data={'kind': kind, **extra}, files={'file': ('fixture.json', json.dumps(rows), 'application/json')})

    def seed(self):
        places = self.import_json('places', [{'place_id': '001', 'name': 'Fixture A'}, {'place_id': '002', 'name': 'Fixture B'}]).json()
        queries = self.import_json('queries', [{'query_id': 'q1', 'query_text': 'fixture', 'query_type': 'semantic', 'split': 'test'}]).json()
        qrels = self.import_json('qrels', [{'query_id': 'q1', 'place_id': '001', 'relevance': 3,'review_status':'confirmed'}, {'query_id': 'q1', 'place_id': '002', 'relevance': 0,'review_status':'confirmed'}], dataset_id=places['id'], queries_id=queries['id']).json()
        return places, queries, qrels

    def test_empty_state_and_templates(self):
        self.assertEqual(self.client.get('/api/datasets').json(), [])
        for kind in data.SCHEMAS:
            for fmt in ('csv', 'xlsx', 'json'):
                self.assertEqual(self.client.get(f'/api/templates/{kind}?format={fmt}').status_code, 200)

    def test_import_all_formats_and_mapping(self):
        rows = [{'id': '001', 'title': 'Bãi biển', 'tags': 'biển;nghỉ dưỡng'}]
        frame = pd.DataFrame(rows)
        xlsx = io.BytesIO()
        frame.to_excel(xlsx, index=False)
        for name, raw in [('places.csv', frame.to_csv(index=False).encode()), ('places.xlsx', xlsx.getvalue()), ('places.json', json.dumps(rows).encode())]:
            response = self.client.post('/api/import', data={'kind':'places','mapping':json.dumps({'place_id':'id','name':'title'})}, files={'file':(name,raw)})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['records'][0]['place_id'], '001')
            self.assertEqual(response.json()['records'][0]['tags'], ['biển', 'nghỉ dưỡng'])
        self.assertEqual(len(self.client.get('/api/datasets').json()), 1)

    def test_duplicates_missing_fractional_and_bad_foreign_keys(self):
        self.assertEqual(self.import_json('places',[{'name':'A'}]).status_code,422)
        self.assertEqual(self.import_json('places',[{'place_id':'1','name':'A'}]*2).status_code,422)
        ds, qs, gt = self.seed()
        for rel in (1.5,4,-1):
            response = self.import_json('qrels',[{'query_id':'q1','place_id':'001','relevance':rel}],dataset_id=ds['id'],queries_id=qs['id'])
            self.assertEqual(response.status_code,422)
        self.assertEqual(self.import_json('qrels',[{'query_id':'unknown','place_id':'001','relevance':1}],dataset_id=ds['id'],queries_id=qs['id']).status_code,422)

    def test_metric_config_validation(self):
        with self.assertRaises(ValueError):
            retrieval.IndexConfig(dataset_id='x', fields=['unknown'])
        with self.assertRaises(ValueError):
            retrieval.IndexConfig(dataset_id='x', method='M1', max_length=10000)

    def test_benchmark_job_export_and_provenance(self):
        ds, qs, gt = self.seed()
        config = retrieval.IndexConfig(dataset_id=ds['id']).model_dump()
        data.save('indexes','lab-test',dict(id='lab-test',created_at=data.now(),config=config,status='ready',indexing_ms=10))
        def fake_search(index_id,query,k,metric,exact):
            return dict(index_id=index_id,method='BM25',query=query,results=[dict(ds['records'][0],rank=1,score=1)],timing=dict(total_ms=2,embedding_ms=0,search_ms=2,model_loading_ms=0,phase='warm'),resources=dict(ram_mb=30))
        body=dict(index_ids=['lab-test'],queries_id=qs['id'],qrels_id=gt['id'],k=1,repeats=2,warmup=1)
        with patch('backend.runner.search',side_effect=fake_search):
            response=self.client.post('/api/benchmarks',json=body)
            self.assertEqual(response.status_code,200,response.text)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                job=self.client.get('/api/jobs/'+response.json()['id']).json()
                if job['status'] not in ('queued','running'):
                    break
                time.sleep(.05)
        self.assertEqual(job['status'],'completed',job)
        run=self.client.get('/api/runs/'+job['result']['run_id']).json()
        self.assertEqual(run['summary'][0]['ndcg'],1)
        self.assertEqual(run['summary'][0]['samples'],2)
        self.assertNotIn('vram_mb',run['summary'][0])
        self.assertEqual(set(retrieval.resources()),{'ram_mb'})
        self.assertIn('versions',run['environment'])
        for fmt in ('csv','xlsx','json','png'):
            response=self.client.get(f"/api/runs/{run['id']}/export?format={fmt}")
            self.assertEqual(response.status_code,200,response.text[:100] if fmt=='json' else fmt)
        # The mocked search is only a software fixture and is confined to a temporary directory.
        data.save('indexes','lab-other',dict(id='lab-other',config={**config,'fields':['name']},status='ready'))
        self.assertEqual(self.client.post('/api/benchmarks',json={**body,'index_ids':['lab-test','lab-other']}).status_code,422)


if __name__=='__main__':
    unittest.main()
