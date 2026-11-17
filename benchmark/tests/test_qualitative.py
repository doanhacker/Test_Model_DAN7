import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import data
from backend.main import app


class QualitativeTests(unittest.TestCase):
    def test_notes_saved_separately_from_ground_truth_and_history_reopens(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            data.save('jobs','search1',dict(id='search1',kind='search',status='completed',created_at='2026-01-01',result=[dict(index_id='idx',method='BM25',query='Manual query',results=[dict(place_id='p',rank=1,score=2)])]))
            with TestClient(app) as client:
                initial=client.get('/api/qualitative/search1/notes')
                self.assertEqual(initial.json()['notes'],[])
                body=dict(observation='Human observation for unit test only',notes=[dict(index_id='idx',place_id='p',verdict='partial',reason='Some conditions missing')])
                saved=client.post('/api/qualitative/search1/notes',json=body)
                self.assertEqual(saved.status_code,200,saved.text)
                self.assertEqual(client.get('/api/qualitative/search1/notes').json()['notes'],body['notes'])
                history=client.get('/api/qualitative/history').json()
                self.assertEqual(history[0]['query'],'Manual query')
                self.assertTrue(history[0]['reviewed'])
                self.assertEqual(data.database_info()['counts']['ground_truth'],0)
                self.assertEqual(data.database_info()['counts']['benchmark_runs'],0)
                self.assertEqual(data.get('jobs','search1')['result'][0]['results'][0]['score'],2)
            with TestClient(app) as restarted:
                self.assertEqual(restarted.get('/api/qualitative/search1/notes').json()['observation'],body['observation'])

    def test_reject_unreturned_place_bad_verdict_and_duplicate_review(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            data.save('jobs','search1',dict(id='search1',kind='search',status='completed',created_at='2026-01-01',result=[dict(index_id='idx',method='BM25',query='Unit test',results=[dict(place_id='p',rank=1)])]))
            data.save('jobs','wrong',dict(id='wrong',kind='benchmark',status='completed',result={}))
            with TestClient(app) as client:
                valid=dict(index_id='idx',place_id='p',verdict='relevant')
                for notes in ([{**valid,'place_id':'unknown'}],[{**valid,'verdict':'invalid'}],[valid,valid]):
                    self.assertEqual(client.post('/api/qualitative/search1/notes',json={'notes':notes}).status_code,422)
                self.assertEqual(client.post('/api/qualitative/wrong/notes',json={}).status_code,422)
