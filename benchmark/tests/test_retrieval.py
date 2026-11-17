import os
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from backend import data, retrieval
from backend.jobs import Cancelled


class DummyJob:
    cancel = threading.Event()
    def check(self):
        pass
    def update(self, *args):
        pass


class RetrievalTests(unittest.TestCase):
    def test_qdrant_real_local_metric_rankings(self):
        """Real Qdrant local engine, synthetic vectors only; no fake model claims."""
        from qdrant_client import QdrantClient, models
        with tempfile.TemporaryDirectory() as temp:
            client=QdrantClient(path=temp)
            vectors=np.array([[1.,0.],[0.,1.],[-1.,0.]])
            for distance in (models.Distance.COSINE,models.Distance.DOT,models.Distance.EUCLID):
                client.create_collection(distance.value,vectors_config=models.VectorParams(size=2,distance=distance))
                client.upsert(distance.value,points=[models.PointStruct(id=i,vector=v.tolist()) for i,v in enumerate(vectors)])
                result=client.query_points(distance.value,query=[1.,0.],limit=3,search_params=models.SearchParams(exact=True)).points
                self.assertEqual([p.id for p in result],[0,1,2])
            client.close()

    def test_bm25_settings_bulk_and_new_identity(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[{'place_id':'1','name':'Biển Mỹ Khê'}],'fixture')
            calls=[]
            def fake_es(method,path,**kwargs):
                calls.append((method,path,kwargs))
                if path.endswith('/_count'):
                    return {'count':1}
                if path.endswith('/_stats/store'):
                    return {'_all':{'primaries':{'store':{'size_in_bytes':10}}}}
                if path=='':
                    return {'version':{'number':'test'}}
                return {}
            with patch.object(retrieval,'es',side_effect=fake_es):
                a=retrieval.build_index(retrieval.IndexConfig(dataset_id=ds['id'],k1=1.2),DummyJob())
                b=retrieval.build_index(retrieval.IndexConfig(dataset_id=ds['id'],k1=1.8),DummyJob())
            self.assertNotEqual(a['id'],b['id'])
            settings=[kwargs['json'] for method,path,kwargs in calls if method=='PUT']
            self.assertEqual(settings[1]['settings']['similarity']['lab_bm25']['k1'],1.8)
            self.assertEqual(settings[0]['mappings']['properties']['name']['search_analyzer'],'lab_vi')


@unittest.skipUnless(os.getenv('RUN_SERVICE_TESTS')=='1','Set RUN_SERVICE_TESTS=1 with Elasticsearch/Qdrant running')
class ServiceTests(unittest.TestCase):
    def test_real_elasticsearch_index_and_query(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[{'place_id':'p1','name':'Biển Mỹ Khê'},{'place_id':'p2','name':'Bảo tàng nghệ thuật'}],'TEST FIXTURE ONLY')
            index=retrieval.build_index(retrieval.IndexConfig(dataset_id=ds['id']),DummyJob())
            try:
                answer=retrieval.search(index['id'],'biển',k=2)
                self.assertEqual(answer['results'][0]['place_id'],'p1')
                self.assertGreater(answer['results'][0]['score'],0)
            finally:
                retrieval.es('DELETE',index['id'])

    def test_real_qdrant_server(self):
        from qdrant_client import models
        client=retrieval.qclient()
        name='test-'+uuid.uuid4().hex
        try:
            client.create_collection(name,vectors_config=models.VectorParams(size=2,distance=models.Distance.COSINE))
            client.upsert(name,points=[models.PointStruct(id=1,vector=[1.,0.]),models.PointStruct(id=2,vector=[0.,1.])],wait=True)
            result=client.query_points(name,query=[1.,0.],limit=2,search_params=models.SearchParams(exact=True)).points
            self.assertEqual(result[0].id,1)
        finally:
            client.delete_collection(name)
            client.close()


@unittest.skipUnless(os.getenv('RUN_MODEL_TESTS')=='1','Set RUN_MODEL_TESTS=1 to download and test embedding checkpoints')
class ModelTests(unittest.TestCase):
    def test_checkpoint_to_qdrant_pipeline(self):
        # Select M1 by default. MODEL_TEST_IDS=M1,M2,M3,M4 tests all, sequentially.
        for method in os.getenv('MODEL_TEST_IDS','M1').split(','):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as temp, patch.object(data,'ROOT',Path(temp)):
                ds=data.import_records('places',[{'place_id':'p1','name':'Biển Mỹ Khê','description':'Tắm biển và ngắm bình minh'},{'place_id':'p2','name':'Bảo tàng nghệ thuật','description':'Tham quan hiện vật'}],'TEST FIXTURE ONLY')
                index=retrieval.build_index(retrieval.IndexConfig(dataset_id=ds['id'],method=method,batch_size=2,max_length=64),DummyJob())
                try:
                    for metric in ('Cosine','Dot','Euclid'):
                        answer=retrieval.search(index['id'],'ngắm bình minh trên biển',k=2,metric=metric)
                        self.assertEqual(len(answer['results']),2)
                        self.assertTrue(all(np.isfinite(p['score']) for p in answer['results']))
                    self.assertEqual(len(index['model']['revision']),40)
                finally:
                    client=retrieval.qclient()
                    for collection in index['collections'].values():
                        client.delete_collection(collection)
                    client.close()
                    retrieval.manager.unload()
