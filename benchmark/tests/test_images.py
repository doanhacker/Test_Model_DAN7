import io
import json
import os
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend import data, database
from backend.main import app
from backend.images import benchmark, encoders, service, store, features, pa1, importer, cnn


def fixture(color,pattern=0):
    image=Image.new('RGB',(128,128),color)
    draw=ImageDraw.Draw(image)
    if pattern:
        draw.rectangle((12,12,65+pattern,95),fill='white')
        draw.line((0,pattern,127,127),fill='black',width=5)
    buffer=io.BytesIO()
    image.save(buffer,format='PNG')
    return buffer.getvalue()


class Job:
    cancel=threading.Event()
    def check(self):
        pass
    def update(self,*args):
        pass


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root_patch=patch.object(data,'ROOT',Path(self.temp.name)/'runtime')
        self.root_patch.start()
        self.client_patch=patch('backend.images.service.retrieval.qclient',side_effect=lambda:QdrantClient(path=str(Path(self.temp.name)/'qdrant')))
        self.client_patch.start()

    def tearDown(self):
        encoders.manager.unload()
        self.client_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def test_inspect_actual_encoder_and_qdrant_vectors(self):
        asset=store.register(fixture('blue',3),'p1')
        cfg=service.IndexConfig(model='handcrafted',device='cpu',batch_size=1)
        encoded=features.extract(asset['id'],cfg,Job())
        built=service.index_images(cfg,[asset['id']],Job())
        indexed=features.indexed(built['index_id'],asset['id'])
        self.assertEqual(encoded['dimension'],2014)
        self.assertEqual(indexed['source'],'qdrant')
        self.assertAlmostEqual(indexed['statistics']['l2_norm'],1,places=5)
        np.testing.assert_allclose(encoded['vector'],indexed['vector'],atol=1e-6)
        self.assertEqual(data.get('image_features',encoded['id'])['image_id'],asset['id'])
        with self.assertRaises(ValueError):
            features.indexed(built['index_id'],'0'*32)

    def test_cnn_gallery_snapshot_is_bounded_and_independent(self):
        a,b,c,q=self.seed()
        extra=store.register(fixture('blue',7),'p1')
        source=service.index_images(service.IndexConfig(),[a['id'],b['id'],c['id'],extra['id']],Job())['metadata']
        ids=cnn.select_gallery(source)
        self.assertEqual(len(ids),3)
        snapshot=cnn.snapshot_baseline(source,ids,'cnn-fixture',Job())
        self.assertEqual(snapshot['indexed_count'],3)
        self.assertEqual(snapshot['gallery_sha256'],data.digest(sorted(ids)))
        self.assertEqual(len(data.get('image_indexes',source['id'])['images']),4)
        client=service.retrieval.qclient()
        try:
            old=client.retrieve(source['id'],ids=[str(uuid.UUID(ids[0]))],with_vectors=True)[0]
            new=client.retrieve(snapshot['id'],ids=[str(uuid.UUID(ids[0]))],with_vectors=True)[0]
            np.testing.assert_allclose(old.vector,new.vector,atol=1e-6)
        finally:
            client.close()

    def test_pa1_blocks_previews_and_score_decomposition(self):
        image=store.decode(fixture('blue',3))
        cfg=encoders.EncoderConfig(weights={'hsv':2,'lbp':0,'glcm':1,'hog':1,'canny':1})
        a=encoders.handcrafted(image,cfg)
        b=encoders.handcrafted(store.decode(fixture('green',8)),cfg)
        blocks=pa1.blocks(a,cfg.model_dump())
        self.assertEqual([x['dimension'] for x in blocks],[128,10,48,1764,64])
        self.assertAlmostEqual(sum(x['energy_share'] for x in blocks),1,places=6)
        self.assertEqual(blocks[1]['energy_share'],0)
        explained=pa1.explain(a,b,cfg.model_dump())
        self.assertAlmostEqual(sum(x['contribution'] for x in explained),float(np.dot(a,b)),places=6)
        self.assertIsNone(explained[1]['cosine'])
        for kind in ('rgb','gray','lbp','glcm','hog','canny'):
            with Image.open(io.BytesIO(pa1.preview(image,kind))) as output:
                self.assertEqual(output.size,(256,256))
        with self.assertRaises(ValueError):
            pa1.preview(image,'unknown')

    def test_pa1_rebuilds_missing_physical_collection(self):
        a,b,c,q=self.seed()
        cfg=service.IndexConfig(groups=['hsv'])
        built=service.index_images(cfg,[a['id'],c['id']],Job())
        client=service.retrieval.qclient()
        client.delete_collection(built['index_id'])
        client.close()
        recovered=service.index_images(cfg,[a['id']],Job())
        self.assertEqual(recovered['metadata']['indexed_count'],2)
        result=service.search_bytes(built['index_id'],fixture('blue',3),10)
        self.assertEqual(len(result['results']),2)
        for row in result['results']:
            self.assertAlmostEqual(sum(x['contribution'] for x in row['feature_scores']),row['score'],places=5)

    def test_evaluate_one_test_query_with_trace(self):
        a,b,c,q=self.seed()
        second=store.register(fixture('orange',9))
        ds=benchmark.create_dataset(benchmark.Dataset(gallery_image_ids=[a['id'],c['id']],
            queries=[benchmark.Query(query_id='q1',image_id=q['id']),benchmark.Query(query_id='q2',image_id=second['id'])],
            judgments=[benchmark.Judgment(query_id='q1',target_id='p1',relevance=3),benchmark.Judgment(query_id='q1',target_id='p2',relevance=0)]))
        index=service.index_images(service.IndexConfig(groups=['hsv']),ds['gallery_image_ids'],Job())
        cfg=benchmark.Config(dataset_id=ds['id'],index_ids=[index['index_id']],objective='visual',repeats=1,warmup=0)
        with self.assertRaises(ValueError):
            benchmark.validate(cfg)
        single=cfg.model_copy(update={'query_ids':['q1']})
        result=data.get('image_runs',benchmark.run(single,Job())['image_run_id'])
        self.assertEqual(len(result['detail']),1)
        self.assertEqual(result['summary'][0]['mrr@10'],1)
        self.assertEqual(result['detail'][0]['query_id'],'q1')
        self.assertEqual(len(result['detail'][0]['evaluation_trace']),2)
        with self.assertRaises(ValueError):
            benchmark.validate(cfg.model_copy(update={'query_ids':['missing']}))

    def test_pa1_ablation_api_same_gallery(self):
        a,b,c,q=self.seed()
        with TestClient(app) as client:
            response=client.post('/api/ai/images/ablation',json={'image_ids':[a['id'],c['id']],
                                                             'config':{'model':'handcrafted','device':'cpu'}})
            self.assertEqual(response.status_code,200,response.text)
            job=self.wait(client,response.json()['id'])
            self.assertEqual(job['status'],'completed',job)
            indexes=job['result']['indexes']
            self.assertEqual(len(indexes),5)
            values=[data.get('image_indexes',i['index_id']) for i in indexes]
            self.assertEqual([i['dimension'] for i in values],[128,138,186,1950,2014])
            self.assertEqual(len({i['gallery_sha256'] for i in values}),1)
            self.assertTrue(all(i['indexed_count']==2 for i in values))

    def test_source_import_replay_and_error_report(self):
        ds=data.import_records('places',[dict(place_id='p1',name='Import fixture'),dict(place_id='p2',name='Broken fixture')],'fixture')
        with database.session(data.ROOT,data.db_path_override()) as conn:
            for n,pid in enumerate(('p1','p2')):
                conn.execute('''INSERT INTO place_images(image_id,dataset_id,place_id,source_url,url_sha256,is_cover,ordinal,source_filename,source_sheet,source_row)
                                VALUES(?,?,?,?,?,?,?,?,?,?)''',(str(n+1)*32,ds['id'],pid,'https://lh3.googleusercontent.com/fixture-'+pid,str(n)*64,True,n,'fixture.xlsx','fixture',n+2))
        def content(identity):
            if identity=='2'*32:
                raise ValueError('source unavailable')
            from fastapi.responses import Response
            return Response(fixture('blue',3),media_type='image/png')
        with patch.object(importer.place_media,'image_content',side_effect=content) as source:
            first=importer.import_photos(ds['id'],None,10,False,Job())
            self.assertEqual((first['imported'],first['failed']),(1,1))
            replay=importer.import_photos(ds['id'],['p1'],10,False,Job())
            self.assertEqual((replay['imported'],replay['reused']),(0,1))
            self.assertEqual(first['image_ids'],replay['image_ids'])
            self.assertEqual(source.call_count,2)
        self.assertEqual(data.get('image_imports',first['id'])['errors'][0]['place_id'],'p2')

    def test_bad_images_and_dedup_pixel_content(self):
        for raw in (b'',b'bad image'):
            with self.assertRaises(ValueError):
                store.decode(raw)
        raw=fixture('red',1)
        asset=store.register(raw,'p1')
        self.assertEqual(store.register(raw,'p1')['id'],asset['id'])
        with self.assertRaises(ValueError):
            store.register(raw,'p2')
        self.assertEqual(store.load(asset['id']).mode,'RGB')

    def test_ablation_vectors_weights_and_registry(self):
        image=store.decode(fixture('blue',4))
        dimensions=[]
        for n in range(1,6):
            config=encoders.EncoderConfig(groups=encoders.GROUPS[:n])
            vector=encoders.handcrafted(image,config)
            self.assertTrue(np.isfinite(vector).all())
            self.assertAlmostEqual(float(np.linalg.norm(vector)),1,places=5)
            dimensions.append(len(vector))
        self.assertEqual(dimensions,[128,138,186,1950,2014])
        a=encoders.handcrafted(image,encoders.EncoderConfig(groups=['hsv','hog']))
        b=encoders.handcrafted(image,encoders.EncoderConfig(groups=['hsv','hog'],weights={'hsv':3,'hog':.2}))
        self.assertFalse(np.allclose(a,b))
        self.assertEqual(len(encoders.MODELS),10)
        with self.assertRaises(ValueError):
            encoders.EncoderConfig(weights={'hsv':-1})

    def seed(self):
        a=store.register(fixture('blue',1),'p1','Biển fixture')
        b=store.register(fixture('blue',2),'p1','Biển fixture')
        c=store.register(fixture('green',10),'p2','Rừng fixture')
        q=store.register(fixture('blue',3))
        return a,b,c,q

    def test_real_handcrafted_qdrant_idempotency_and_delete(self):
        a,b,c,q=self.seed()
        cfg=service.IndexConfig(groups=['hsv','lbp'])
        result=service.index_images(cfg,[a['id'],b['id'],c['id']],Job())
        again=service.index_images(cfg,[a['id'],b['id']],Job())
        self.assertEqual(result['index_id'],again['index_id'])
        self.assertEqual(again['added'],0)
        answer=service.search_bytes(result['index_id'],fixture('blue',3),10)
        self.assertEqual(len(answer['results']),2)
        self.assertEqual(answer['results'][0]['place_id'],'p1')
        raw=service.search_bytes(result['index_id'],fixture('blue',3),10,deduplicate=False)
        self.assertEqual(len(raw['results']),3)
        service.delete_image(result['index_id'],a['id'])
        self.assertNotIn(a['id'],data.get('image_indexes',result['index_id'])['images'])
        replay=service.index_images(cfg,[a['id']],Job())
        self.assertEqual(replay['added'],1)

    def test_retry_after_collection_creation_failure(self):
        asset=store.register(fixture('blue',2),'p1')
        config=service.IndexConfig(groups=['hsv'])
        with patch.object(QdrantClient,'create_collection',side_effect=RuntimeError('temporary service outage')):
            with self.assertRaises(RuntimeError):
                service.index_images(config,[asset['id']],Job())
        self.assertEqual(data.listing('image_indexes')[0]['status'],'failed')
        result=service.index_images(config,[asset['id']],Job())
        self.assertEqual(result['metadata']['status'],'ready')
        self.assertEqual(len(service.search_bytes(result['index_id'],fixture('blue',3),1)['results']),1)

    def test_dataset_leakage_and_missing_visual_labels(self):
        a,b,c,q=self.seed()
        with self.assertRaises(ValueError):
            benchmark.create_dataset(benchmark.Dataset(gallery_image_ids=[a['id']],queries=[benchmark.Query(query_id='q',image_id=a['id'])]))
        ds=benchmark.create_dataset(benchmark.Dataset(gallery_image_ids=[a['id'],c['id']],queries=[benchmark.Query(query_id='q',image_id=q['id'],exact_place_ids=['p1'])]))
        index=service.index_images(service.IndexConfig(groups=['hsv']),ds['gallery_image_ids'],Job())
        with self.assertRaises(ValueError):
            benchmark.validate(benchmark.Config(dataset_id=ds['id'],index_ids=[index['index_id']]))
        config=benchmark.Config(dataset_id=ds['id'],index_ids=[index['index_id']],objective='exact',repeats=1,warmup=0)
        run=benchmark.run(config,Job())
        result=data.get('image_runs',run['image_run_id'])
        self.assertEqual(result['summary'][0]['recall@1'],1)
        self.assertNotIn('ndcg@10',result['summary'][0])

    def test_full_real_benchmark_both_objectives(self):
        a,b,c,q=self.seed()
        ds=benchmark.create_dataset(benchmark.Dataset(gallery_image_ids=[a['id'],b['id'],c['id']],
            queries=[benchmark.Query(query_id='q1',image_id=q['id'],exact_place_ids=['p1'])],
            judgments=[benchmark.Judgment(query_id='q1',target_id='p1',relevance=3),benchmark.Judgment(query_id='q1',target_id='p2',relevance=1)]))
        ids=[]
        for groups in (['hsv'],['hsv','lbp']):
            ids.append(service.index_images(service.IndexConfig(groups=groups),ds['gallery_image_ids'],Job())['index_id'])
        run=benchmark.run(benchmark.Config(dataset_id=ds['id'],index_ids=ids,repeats=2,warmup=1),Job())
        result=data.get('image_runs',run['image_run_id'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(result['summary']),4)
        self.assertEqual(len(result['detail']),8)
        self.assertEqual(result['summary'][1]['ndcg@10'],1)
        self.assertEqual(result['summary'][1]['precision@10'],.2)
        self.assertEqual(result['dataset']['judgments'][1]['relevance'],1)

    def test_visual_image_target_and_provisional_missing(self):
        a,b,c,q=self.seed()
        ds=benchmark.create_dataset(benchmark.Dataset(gallery_image_ids=[a['id'],c['id']],
            queries=[benchmark.Query(query_id='q',image_id=q['id'])],visual_target='image',
            judgments=[benchmark.Judgment(query_id='q',target_id=a['id'],relevance=3)]))
        index=service.index_images(service.IndexConfig(groups=['hsv']),ds['gallery_image_ids'],Job())
        config=benchmark.Config(dataset_id=ds['id'],index_ids=[index['index_id']],objective='visual',official=False,repeats=1,warmup=0)
        result=data.get('image_runs',benchmark.run(config,Job())['image_run_id'])
        self.assertIsNone(result['summary'][0]['ndcg@10'])
        self.assertEqual(result['detail'][0]['status'],'incomplete_labels')
        self.assertFalse(result['samples'][0]['deduplicate'])

    def test_api_upload_background_search_and_validation(self):
        with TestClient(app) as client:
            models=client.get('/api/ai/images/models')
            self.assertEqual(models.status_code,200)
            self.assertEqual(client.get('/image-search').status_code,200)
            bad=client.post('/api/ai/images/assets',files={'file':('bad.jpg',b'invalid','image/jpeg')})
            self.assertEqual(bad.status_code,422)
            malformed=client.post('/api/ai/images/index',data={'place_id':'p1','config':'[]'},files={'file':('fixture.png',fixture('blue',1),'image/png')})
            self.assertEqual(malformed.status_code,422)
            response=client.post('/api/ai/images/index',data={'place_id':'p1','config':json.dumps({'groups':['hsv']})},files={'file':('fixture.png',fixture('blue',1),'image/png')})
            self.assertEqual(response.status_code,200,response.text)
            job=self.wait(client,response.json()['id'])
            self.assertEqual(job['status'],'completed',job)
            index_id=job['result']['index_id']
            response=client.post('/api/ai/images/search',data={'index_id':index_id},files={'file':('query.png',fixture('blue',2),'image/png')})
            answer=self.wait(client,response.json()['id'])
            self.assertEqual(answer['status'],'completed',answer)
            self.assertEqual(answer['result']['results'][0]['place_id'],'p1')
            self.assertEqual(client.get('/api/indexes').json(),[]) # text indexes untouched

    def wait(self,client,key):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            job=client.get('/api/jobs/'+key).json()
            if job['status'] not in ('queued','running'):
                return job
            time.sleep(.05)
        self.fail('Job timed out')


@unittest.skipUnless(os.getenv('RUN_IMAGE_MODEL_TESTS')=='1','Set RUN_IMAGE_MODEL_TESTS=1 for real pretrained weights')
class PretrainedTests(unittest.TestCase):
    def test_real_checkpoint_outputs(self):
        for name in os.getenv('IMAGE_TEST_MODELS','mobilenet_v3_large,clip_b32').split(','):
            with self.subTest(model=name):
                cfg=encoders.EncoderConfig(model=name,device='cpu',batch_size=2)
                try:
                    encoders.manager.load(cfg)
                    vectors,timing=encoders.manager.encode([store.decode(fixture('blue',1)),store.decode(fixture('green',10))],cfg)
                    self.assertEqual(vectors.shape,(2,encoders.MODELS[name]['dimension']))
                    self.assertTrue(np.isfinite(vectors).all())
                    np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
                finally:
                    encoders.manager.unload()


@unittest.skipUnless(os.getenv('RUN_IMAGE_SERVICE_TESTS')=='1','Set RUN_IMAGE_SERVICE_TESTS=1 with Qdrant server running')
class ImageServiceTests(unittest.TestCase):
    def test_real_qdrant_server_pipeline(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(data,'ROOT',Path(temp)/'runtime'):
            a=store.register(fixture('blue',1),'p1','Test fixture beach')
            b=store.register(fixture('green',4),'p2','Test fixture forest')
            q=store.register(fixture('blue',2))
            ds=benchmark.create_dataset(benchmark.Dataset(gallery_image_ids=[a['id'],b['id']],
                queries=[benchmark.Query(query_id='q1',image_id=q['id'],exact_place_ids=['p1'])],
                judgments=[benchmark.Judgment(query_id='q1',target_id='p1',relevance=3),benchmark.Judgment(query_id='q1',target_id='p2',relevance=1)]))
            config=service.IndexConfig(collection='test-'+uuid.uuid4().hex,groups=['hsv','lbp'])
            result=service.index_images(config,ds['gallery_image_ids'],Job())
            try:
                answer=service.search_bytes(result['index_id'],fixture('blue',2),2)
                self.assertEqual(answer['results'][0]['place_id'],'p1')
                run=benchmark.run(benchmark.Config(dataset_id=ds['id'],index_ids=[result['index_id']],repeats=1,warmup=1),Job())
                with TestClient(app) as client:
                    for fmt in ('json','csv'):
                        response=client.get(f"/api/ai/images/runs/{run['image_run_id']}/export?format={fmt}")
                        self.assertEqual(response.status_code,200)
            finally:
                client=service.retrieval.qclient()
                client.delete_collection(result['index_id'])
                client.close()
                encoders.manager.unload()
