import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import data,retrieval
from backend.main import app


class QueryEvaluationTests(unittest.TestCase):
    def test_confirmed_query_metrics_match_hand_calculation_and_persist(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[dict(place_id=p,name='Unit '+p) for p in ('a','b','c','d')],'unit test')
            qs=data.import_records('queries',[dict(query_id='q',query_text='Unit query',query_type='unit',split='test')],'unit test')
            gt=data.import_records('qrels',[dict(query_id='q',place_id=p,relevance=r,review_status='confirmed',reason='Unit test') for p,r in [('a',3),('b',0),('c',2)]],'unit test',ds['id'],qs['id'])
            data.save('indexes','idx',dict(id='idx',status='ready',config=retrieval.IndexConfig(dataset_id=ds['id']).model_dump()))
            data.save('jobs','job',dict(id='job',kind='search',status='completed',created_at=data.now(),result=[dict(index_id='idx',method='BM25',query='Unit query',k=3,
                results=[dict(place_id=p,rank=i+1,score=3-i) for i,p in enumerate(['b','a','c'])],timing=dict(total_ms=20,embedding_ms=0,search_ms=20,model_loading_ms=0),resources=dict(ram_mb=100))]))
            with TestClient(app) as client:
                body=dict(job_id='job',query_id='q',qrels_id=gt['id'],k=2,threshold=1,official=False)
                response=client.post('/api/evaluation/query',json=body)
                self.assertEqual(response.status_code,200,response.text)
                result=response.json()
                metric=result['detail'][0]
                self.assertEqual(metric['recall'],.5)
                self.assertEqual(metric['mrr'],.5)
                self.assertAlmostEqual(metric['ndcg'],(7/math.log2(3))/(7+3/math.log2(3)))
                self.assertEqual(metric['recall_status'],'pooled')
                self.assertEqual(metric['mrr_status'],'exact')
                stored=data.get('runs',result['id'])
                self.assertTrue(stored['evaluation_only'])
                self.assertEqual(stored['config']['qrels_id'],gt['id'])
                self.assertEqual(data.database_info()['counts']['run_metrics'],1)
                repeated=client.post('/api/database/runs/'+result['id']+'/rerun').json()
                deadline=time.monotonic()+15
                while time.monotonic()<deadline:
                    repeated_job=client.get('/api/jobs/'+repeated['id']).json()
                    if repeated_job['status'] not in ('queued','running'):
                        break
                    time.sleep(.05)
                self.assertEqual(repeated_job['status'],'completed',repeated_job)
                repeated_run=data.get('runs',repeated_job['result']['run_id'])
                self.assertEqual(len(repeated_run['query_set']),1)
                self.assertEqual(repeated_run['config']['query_id'],'q')
                exact=client.post('/api/evaluation/query',json={**body,'official':True}).json()['detail'][0]
                self.assertIsNone(exact['recall'])
                self.assertIsNone(exact['ndcg'])
                self.assertEqual(exact['mrr'],.5)
                partial=data.import_records('qrels',[dict(query_id='q',place_id='a',relevance=3,review_status='confirmed'),dict(query_id='q',place_id='b',relevance=0,review_status='confirmed'),dict(query_id='q',place_id='c',relevance=None,review_status='pending')],'partial pool',ds['id'],qs['id'])
                partial_result=client.post('/api/evaluation/query',json={**body,'qrels_id':partial['id']}).json()['detail'][0]
                self.assertIsNone(partial_result['recall'])
                self.assertIsNone(partial_result['ndcg'])
                self.assertEqual(partial_result['mrr'],.5)
                self.assertEqual(partial_result['missing_pool'],1)
                self.assertEqual(client.post('/api/evaluation/query',json={**body,'k':4}).status_code,422)
                self.assertEqual(client.post('/api/evaluation/query',json={**body,'query_id':'wrong'}).status_code,422)

    def test_pending_labels_do_not_create_quality_scores(self):
        from backend.metrics import evaluate_query
        result=evaluate_query(['a','b'],{},['a','b'],k=2,official=False)
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['mrr'])
        self.assertIsNone(result['ndcg'])
