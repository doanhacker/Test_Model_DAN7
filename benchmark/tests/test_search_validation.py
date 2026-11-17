import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend import data,retrieval
from backend.main import app


class SearchValidationTests(unittest.TestCase):
    def test_bm25_accepts_empty_or_irrelevant_metric(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[dict(place_id='p',name='Validation fixture')],'unit test')
            cfg=retrieval.IndexConfig(dataset_id=ds['id']).model_dump()
            data.save('indexes','idx',dict(id='idx',status='ready',config=cfg))
            with patch.object(retrieval,'es',return_value={'hits':{'hits':[{'_id':'p','_score':1}]}}):
                for metric in ('',None,'BM25','Cosine'):
                    answer=retrieval.search('idx','query',10,metric)
                    self.assertEqual(answer['results'][0]['place_id'],'p')
                    self.assertEqual(answer['score_type'],'BM25')
                    self.assertIsNone(answer['metric'])

    def test_precise_validation_for_dense_query_and_k(self):
        self.assertEqual(retrieval.normalize_metric(''),'Cosine')
        self.assertEqual(retrieval.normalize_metric(' euclidean '),'Euclid')
        with self.assertRaisesRegex(ValueError,'Similarity'):
            retrieval.normalize_metric('wrong')
        with self.assertRaisesRegex(ValueError,'khoảng trắng'):
            retrieval.search('unused','  ',10)
        with self.assertRaisesRegex(ValueError,'Top-K'):
            retrieval.search('unused','query',0)
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)),TestClient(app) as client:
            response=client.post('/api/search',json=dict(index_ids=['missing'],query='  ',k=10,metric=''))
            self.assertEqual(response.status_code,422)
            response=client.post('/api/search',json=dict(index_ids=['missing'],query='query',k=0))
            self.assertEqual(response.status_code,422)
