import math
import unittest
from backend.metrics import evaluate_query, latency_summary


class MetricTests(unittest.TestCase):
    def test_mrr_exact_from_known_prefix_without_full_corpus(self):
        result=evaluate_query(['a','b','c'],{'a':0,'b':3},['a','b','c','d'],k=3)
        self.assertEqual(result['mrr'],.5)
        self.assertEqual(result['mrr_status'],'exact')
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['ndcg'])
        self.assertEqual(result['missing_top_k'],1)
        self.assertIsNone(result['trace'][2]['relevance'])

    def test_unknown_prefix_blocks_mrr(self):
        result=evaluate_query(['a','b'],{'b':3},['a','b'],k=2,official=False)
        self.assertIsNone(result['mrr'])
        self.assertEqual(result['mrr_status'],'needs_prefix_labels')

    def test_pool_values_have_explicit_scope(self):
        result=evaluate_query(['a','b'],{'a':3,'b':0,'c':2},['a','b','c','d'],k=2,official=False)
        self.assertEqual(result['recall'],.5)
        self.assertEqual(result['recall_status'],'pooled')
        self.assertEqual(result['ndcg_status'],'pooled')
        self.assertEqual(result['mrr'],1)
    def test_hand_calculation(self):
        result = evaluate_query(['b', 'c', 'a'], {'a': 3, 'b': 0, 'c': 2, 'd': 1}, ['a', 'b', 'c', 'd'], k=3, threshold=2)
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['mrr'], .5)
        expected = (3 / math.log2(3) + 7 / 2) / (7 + 3 / math.log2(3) + 1 / 2)
        self.assertAlmostEqual(result['ndcg'], expected)

    def test_missing_labels(self):
        for official in (True, False):
            result = evaluate_query(['a', 'b'], {'a': 3}, ['a', 'b'], official=official)
            self.assertEqual(result['status'], 'incomplete_labels')
            self.assertIsNone(result['recall'])

    def test_official_requires_full_corpus(self):
        self.assertIsNone(evaluate_query(['a'], {'a': 3}, ['a', 'b'], k=1)['ndcg'])
        self.assertEqual(evaluate_query(['a'], {'a': 3}, ['a', 'b'], k=1, official=False)['status'], 'provisional')

    def test_no_relevant(self):
        result = evaluate_query(['a'], {'a': 0}, ['a'], k=1)
        self.assertEqual(result['status'], 'no_relevant')
        self.assertIsNone(result['recall'])
        self.assertEqual(result['mrr'], 0)
        self.assertEqual(result['ndcg'], 0)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_query(['a', 'a'], {'a': 3}, ['a'])

    def test_no_hits_and_short_results(self):
        result = evaluate_query([], {'a': 3}, ['a'], k=10)
        self.assertEqual(result['precision'], 0)
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['mrr'], 0)
        result = evaluate_query(['a'], {'a': 3}, ['a'], k=10)
        self.assertEqual(result['precision'], .1)
        self.assertEqual(result['recall'], 1)

    def test_threshold_and_rank(self):
        result = evaluate_query(['a','b'], {'a': 1, 'b': 3}, ['a','b'], k=1, threshold=2)
        self.assertEqual(result['mrr'], 0)
        self.assertAlmostEqual(result['ndcg'], 1/7)

    def test_latency(self):
        result = latency_summary([10, 20, 30, 40, 50])
        self.assertEqual(result['average_ms'], 30)
        self.assertEqual(result['p50_ms'], 30)
        self.assertEqual(result['p95_ms'], 48)
        self.assertEqual(result['p99_ms'], 49.6)
        self.assertIsNone(latency_summary([])['p95_ms'])


if __name__ == '__main__':
    unittest.main()
