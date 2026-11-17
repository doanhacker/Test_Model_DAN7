import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import data, real_data, retrieval, runner


class RealDataTests(unittest.TestCase):
    def test_supplied_workbook_import_and_100_grounded_queries(self):
        path=Path(__file__).resolve().parents[2]/'data'/'danang_merged_serpapi_da_lam_sach.xlsx'
        raw=path.read_bytes()
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            result=real_data.import_excel(raw,path.name)
            self.assertEqual(result['valid_places'],6052)
            self.assertEqual(result['excluded_city'],259)
            self.assertEqual(result['merged_duplicate_rows'],1)
            ds=data.get('datasets',result['dataset_id'])
            report=data.get('source_reports',ds['id'])
            self.assertEqual(len({r['place_id'] for r in ds['records']}),6052)
            self.assertEqual(sum(len(v) for v in report['source_links'].values()),6053)
            created=real_data.generate_queries(ds['id'])
            self.assertEqual(set(created['groups'].values()),{20})
            qs=data.get('datasets',created['queries_id'])
            self.assertEqual(len(qs['records']),100)
            names={r['name'] for r in ds['records']}
            self.assertTrue(all(q['query_text'] in names for q in qs['records'] if q['query_type']=='ten_rieng'))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),result['source_sha256'])

    def test_only_confirmed_labels_enter_metrics(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[{'place_id':'p','name':'Unit test only'}],'test')
            qs=data.import_records('queries',[{'query_id':'q','query_text':'test','query_type':'test','split':'test'}],'test')
            cfg=retrieval.IndexConfig(dataset_id=ds['id']).model_dump()
            data.save('indexes','test',{'id':'test','config':cfg,'status':'ready'})
            for status in ('pending','proposed','confirmed'):
                gt=data.import_records('qrels',[{'query_id':'q','place_id':'p','relevance':3,'reason':'unit test only','review_status':status}],'test',ds['id'],qs['id'])
                body=runner.BenchmarkConfig(index_ids=['test'],queries_id=qs['id'],qrels_id=gt['id'])
                if status=='confirmed':
                    self.assertEqual(runner.validate_config(body)[5],{'q':{'p':3}})
                else:
                    with self.assertRaisesRegex(ValueError,'thiếu'):
                        runner.validate_config(body)
                self.assertEqual(real_data.coverage(ds['id'],qs['id'],gt['id'])['confirmed'],int(status=='confirmed'))

    def test_pending_empty_relevance_and_flatten_nested_amenities(self):
        self.assertEqual(real_data.flatten("{'accessibility':['Lối vào cho xe lăn'],'amenities':['Nhà vệ sinh']}"),['Lối vào cho xe lăn','Nhà vệ sinh'])
        with tempfile.TemporaryDirectory() as temp,patch.object(data,'ROOT',Path(temp)):
            ds=data.import_records('places',[{'place_id':'p','name':'Unit only'},{'place_id':'p2','name':'Unit only 2'}],'test')
            qs=data.import_records('queries',[{'query_id':'q','query_text':'test','query_type':'test','split':'test'}],'test')
            gt=data.import_records('qrels',[{'query_id':'q','place_id':'p','relevance':None,'review_status':'pending'},{'query_id':'q','place_id':'p2','relevance':2,'review_status':'confirmed'}],'test',ds['id'],qs['id'])
            self.assertIsNone(gt['records'][0]['relevance'])
            self.assertEqual(gt['records'][1]['relevance'],2)
