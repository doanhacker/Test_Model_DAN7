"""Wait for a real indexing job, then smoke-test all three vector distances.

Usage: python scripts/verify_dense_job.py JOB_ID
Writes runtime/dense-verification-JOB_ID.json; never assigns relevance labels.
"""
import json
import math
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def wait(client, identity):
    while True:
        response = client.get('/api/jobs/' + identity)
        response.raise_for_status()
        job = response.json()
        if job['status'] not in ('queued', 'running'):
            if job['status'] != 'completed':
                raise RuntimeError(job.get('error', job['status']))
            return job['result']
        time.sleep(5)


def main(identity):
    report = {'indexing_job_id': identity, 'status': 'running', 'checks': []}
    target = ROOT / 'runtime' / ('dense-verification-' + identity + '.json')
    try:
        with httpx.Client(base_url='http://127.0.0.1:8000', timeout=120) as client:
            index = wait(client, identity)
            if index['status'] != 'ready' or index['indexed_count'] != index['count']:
                raise RuntimeError('Index is not complete')
            report['index_id'] = index['id']
            report['indexed_count'] = index['indexed_count']
            prepared = client.get('/api/real-data/preparation').json()
            queries = client.get('/api/datasets/' + prepared['queries']['queries_id']).json()
            query = queries['records'][0]['query_text']
            for metric in ('Cosine', 'Dot', 'Euclid'):
                response = client.post('/api/search', json={'index_ids': [index['id']], 'query': query,
                                                           'k': 10, 'metric': metric, 'exact': True})
                response.raise_for_status()
                answers = wait(client, response.json()['id'])
                answer = answers[0]
                if len(answer['results']) != min(10, index['count']):
                    raise RuntimeError('Unexpected result count')
                if not all(math.isfinite(row['score']) for row in answer['results']):
                    raise RuntimeError('Non-finite score')
                report['checks'].append(answer)
            report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
    finally:
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main(sys.argv[1])
