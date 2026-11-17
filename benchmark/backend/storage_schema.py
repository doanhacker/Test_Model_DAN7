"""Shared PostgreSQL schema and typed snapshot projections."""

import json

from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(
 group_name TEXT NOT NULL,record_id TEXT NOT NULL,created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL,payload TEXT NOT NULL CHECK(payload::jsonb IS NOT NULL),
 PRIMARY KEY(group_name,record_id));
CREATE INDEX IF NOT EXISTS documents_by_date ON documents(group_name,created_at DESC,record_id);
CREATE TABLE IF NOT EXISTS dataset_versions(
 id TEXT PRIMARY KEY,kind TEXT NOT NULL,name TEXT NOT NULL,sha256 TEXT,record_count INTEGER NOT NULL,
 parent_dataset_id TEXT,parent_queries_id TEXT,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS places(
 dataset_id TEXT NOT NULL,place_id TEXT NOT NULL,name TEXT NOT NULL,category TEXT NOT NULL,
 city TEXT NOT NULL,address TEXT NOT NULL,description TEXT NOT NULL,tags_json TEXT NOT NULL,
 amenities_json TEXT NOT NULL,ordinal INTEGER NOT NULL,
 PRIMARY KEY(dataset_id,place_id),
 FOREIGN KEY(dataset_id) REFERENCES dataset_versions(id) DEFERRABLE INITIALLY DEFERRED);
CREATE INDEX IF NOT EXISTS places_by_name ON places(dataset_id,name);
CREATE TABLE IF NOT EXISTS place_images(
 image_id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL,place_id TEXT NOT NULL,
 source_url TEXT NOT NULL,url_sha256 TEXT NOT NULL,is_cover BOOLEAN NOT NULL DEFAULT FALSE,
 ordinal INTEGER NOT NULL,source_filename TEXT NOT NULL,source_sheet TEXT NOT NULL,source_row INTEGER NOT NULL,
 image_content BYTEA,mime_type TEXT,content_sha256 TEXT,status TEXT NOT NULL DEFAULT 'pending',
 last_error TEXT,updated_at TEXT,
 UNIQUE(dataset_id,place_id,url_sha256),
 FOREIGN KEY(dataset_id,place_id) REFERENCES places(dataset_id,place_id) DEFERRABLE INITIALLY DEFERRED);
CREATE INDEX IF NOT EXISTS images_by_place ON place_images(dataset_id,place_id,is_cover DESC,ordinal);
CREATE TABLE IF NOT EXISTS queries(
 queries_id TEXT NOT NULL,query_id TEXT NOT NULL,query_text TEXT NOT NULL,
 query_type TEXT NOT NULL,split TEXT NOT NULL,ordinal INTEGER NOT NULL,
 PRIMARY KEY(queries_id,query_id),
 FOREIGN KEY(queries_id) REFERENCES dataset_versions(id) DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS ground_truth(
 qrels_id TEXT NOT NULL,query_id TEXT NOT NULL,place_id TEXT NOT NULL,
 dataset_id TEXT NOT NULL,queries_id TEXT NOT NULL,relevance INTEGER CHECK(relevance BETWEEN 0 AND 3),
 reason TEXT NOT NULL,review_status TEXT NOT NULL CHECK(review_status IN ('pending','proposed','confirmed')),
 CHECK(review_status!='confirmed' OR relevance IS NOT NULL),
 PRIMARY KEY(qrels_id,query_id,place_id),
 FOREIGN KEY(qrels_id) REFERENCES dataset_versions(id) DEFERRABLE INITIALLY DEFERRED,
 FOREIGN KEY(dataset_id,place_id) REFERENCES places(dataset_id,place_id) DEFERRABLE INITIALLY DEFERRED,
 FOREIGN KEY(queries_id,query_id) REFERENCES queries(queries_id,query_id) DEFERRABLE INITIALLY DEFERRED);
CREATE INDEX IF NOT EXISTS ground_truth_by_query ON ground_truth(qrels_id,query_id,review_status);
CREATE TABLE IF NOT EXISTS source_links(
 dataset_id TEXT NOT NULL,place_id TEXT NOT NULL,sheet TEXT NOT NULL,source_row INTEGER NOT NULL,
 source_place_id TEXT NOT NULL,filename TEXT NOT NULL,source_sha256 TEXT NOT NULL,
 PRIMARY KEY(dataset_id,place_id,sheet,source_row),
 FOREIGN KEY(dataset_id,place_id) REFERENCES places(dataset_id,place_id) DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS search_indexes(
 id TEXT PRIMARY KEY,dataset_id TEXT NOT NULL,method TEXT NOT NULL,status TEXT NOT NULL,
 indexed_count INTEGER,config_json TEXT NOT NULL,created_at TEXT NOT NULL,
 FOREIGN KEY(dataset_id) REFERENCES dataset_versions(id) DEFERRABLE INITIALLY DEFERRED);
CREATE TABLE IF NOT EXISTS benchmark_runs(
 run_kind TEXT NOT NULL,id TEXT NOT NULL,dataset_id TEXT,queries_id TEXT,qrels_id TEXT,
 status TEXT NOT NULL,created_at TEXT NOT NULL,finished_at TEXT,config_json TEXT NOT NULL,
 environment_json TEXT NOT NULL,error TEXT,
 PRIMARY KEY(run_kind,id));
CREATE INDEX IF NOT EXISTS runs_by_date ON benchmark_runs(created_at DESC);
CREATE TABLE IF NOT EXISTS run_metrics(
 run_kind TEXT NOT NULL,run_id TEXT NOT NULL,index_id TEXT NOT NULL,objective TEXT NOT NULL,
 method TEXT NOT NULL,k INTEGER,query_count INTEGER,sample_count INTEGER,eligible_queries INTEGER,
 recall_at_k DOUBLE PRECISION,mrr_at_k DOUBLE PRECISION,ndcg_at_k DOUBLE PRECISION,precision_at_k DOUBLE PRECISION,
 recall_at_1 DOUBLE PRECISION,recall_at_5 DOUBLE PRECISION,average_ms DOUBLE PRECISION,p50_ms DOUBLE PRECISION,p95_ms DOUBLE PRECISION,p99_ms DOUBLE PRECISION,
 ram_mb DOUBLE PRECISION,embedding_ms DOUBLE PRECISION,search_ms DOUBLE PRECISION,indexing_ms DOUBLE PRECISION,model_loading_ms DOUBLE PRECISION,
 payload TEXT NOT NULL,
 PRIMARY KEY(run_kind,run_id,index_id,objective),
 FOREIGN KEY(run_kind,run_id) REFERENCES benchmark_runs(run_kind,id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS query_metrics(
 run_kind TEXT NOT NULL,run_id TEXT NOT NULL,index_id TEXT NOT NULL,query_id TEXT NOT NULL,
 repeat INTEGER NOT NULL,objective TEXT NOT NULL,query_type TEXT,status TEXT,
 recall_at_k DOUBLE PRECISION,mrr_at_k DOUBLE PRECISION,ndcg_at_k DOUBLE PRECISION,precision_at_k DOUBLE PRECISION,
 total_ms DOUBLE PRECISION,embedding_ms DOUBLE PRECISION,search_ms DOUBLE PRECISION,ram_mb DOUBLE PRECISION,payload TEXT NOT NULL,
 PRIMARY KEY(run_kind,run_id,index_id,query_id,repeat,objective),
 FOREIGN KEY(run_kind,run_id) REFERENCES benchmark_runs(run_kind,id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS search_results(
 run_kind TEXT NOT NULL,run_id TEXT NOT NULL,index_id TEXT NOT NULL,query_id TEXT NOT NULL,
 repeat INTEGER NOT NULL,objective TEXT NOT NULL,rank INTEGER NOT NULL,
 place_id TEXT,image_id TEXT,score DOUBLE PRECISION,payload TEXT NOT NULL,
 PRIMARY KEY(run_kind,run_id,index_id,query_id,repeat,objective,rank),
 FOREIGN KEY(run_kind,run_id) REFERENCES benchmark_runs(run_kind,id) ON DELETE CASCADE);
CREATE OR REPLACE VIEW latest_places AS
 SELECT * FROM places WHERE dataset_id=(SELECT id FROM dataset_versions WHERE kind='places' ORDER BY created_at DESC,id LIMIT 1);
CREATE OR REPLACE VIEW benchmark_history AS
 SELECT r.run_kind,r.id AS run_id,r.dataset_id,r.status,r.created_at,r.finished_at,
 m.index_id,m.method,m.objective,m.k,m.query_count,m.sample_count,
 m.recall_at_k,m.mrr_at_k,m.ndcg_at_k,m.precision_at_k,m.average_ms,m.p50_ms,m.p95_ms,m.p99_ms,m.ram_mb
 FROM benchmark_runs r LEFT JOIN run_metrics m ON r.run_kind=m.run_kind AND r.id=m.run_id;
"""

TABLES=('dataset_versions','places','place_images','queries','ground_truth','source_links','search_indexes',
        'benchmark_runs','run_metrics','query_metrics','search_results','documents')

def encode(value):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def put_document(conn,group,key,value):
    timestamp=datetime.now(timezone.utc).isoformat()
    conn.execute('INSERT INTO documents VALUES(?,?,?,?,?) ON CONFLICT(group_name,record_id) DO UPDATE SET created_at=excluded.created_at,updated_at=excluded.updated_at,payload=excluded.payload',
                 (group,key,value.get('created_at',''),timestamp,encode(value)))
    project(conn,group,key,value)

def project(conn,group,key,value):
    """Typed tables and full reproducibility document are updated in one transaction."""
    if group=='datasets':
        records=value['records']
        conn.execute('INSERT INTO dataset_versions VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,sha256=excluded.sha256,record_count=excluded.record_count',
                     (key,value['kind'],value.get('name',key),value.get('sha256'),len(records),value.get('dataset_id'),value.get('queries_id'),value.get('created_at','')))
        if value['kind']=='places':
            conn.execute('DELETE FROM places WHERE dataset_id=?',(key,))
            conn.executemany('INSERT INTO places VALUES(?,?,?,?,?,?,?,?,?,?)',
                [(key,r['place_id'],r['name'],r.get('category',''),r.get('city',''),r.get('address',''),r.get('description',''),encode(r.get('tags',[])),encode(r.get('amenities',[])),i) for i,r in enumerate(records)])
        elif value['kind']=='queries':
            conn.execute('DELETE FROM queries WHERE queries_id=?',(key,))
            conn.executemany('INSERT INTO queries VALUES(?,?,?,?,?,?)',[(key,r['query_id'],r['query_text'],r['query_type'],r['split'],i) for i,r in enumerate(records)])
        elif value['kind']=='qrels':
            conn.execute('DELETE FROM ground_truth WHERE qrels_id=?',(key,))
            conn.executemany('INSERT INTO ground_truth VALUES(?,?,?,?,?,?,?,?)',
                [(key,r['query_id'],r['place_id'],value['dataset_id'],value['queries_id'],r.get('relevance'),r.get('reason',''),r.get('review_status','confirmed')) for r in records])
    elif group=='source_reports':
        conn.execute('DELETE FROM source_links WHERE dataset_id=?',(key,))
        conn.executemany('INSERT INTO source_links VALUES(?,?,?,?,?,?,?)',
            [(key,pid,link['sheet'],link['row'],link.get('source_place_id',pid),value['filename'],value['source_sha256']) for pid,links in value['source_links'].items() for link in links])
    elif group=='indexes':
        cfg=value['config']
        conn.execute('INSERT INTO search_indexes VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,indexed_count=excluded.indexed_count,config_json=excluded.config_json',
                     (key,cfg['dataset_id'],cfg['method'],value['status'],value.get('indexed_count',value.get('count')),encode(cfg),value.get('created_at','')))
    elif group in ('runs','image_runs'):
        kind='text' if group=='runs' else 'image'
        cfg=value.get('config',{})
        conn.execute('INSERT INTO benchmark_runs VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_kind,id) DO UPDATE SET status=excluded.status,finished_at=excluded.finished_at,config_json=excluded.config_json,environment_json=excluded.environment_json,error=excluded.error',
                     (kind,key,value.get('dataset_id',cfg.get('dataset_id')),cfg.get('queries_id'),cfg.get('qrels_id'),value['status'],value.get('created_at',''),value.get('finished_at'),encode(cfg),encode(value.get('environment',{})),value.get('error')))
        for table in ('run_metrics','query_metrics','search_results'):
            conn.execute(f'DELETE FROM {table} WHERE run_kind=? AND run_id=?',(kind,key))
        for row in value.get('summary',[]):
            metrics=(kind,key,row['index_id'],row.get('objective','text'),row.get('method',row.get('model','')),
                     row.get('k',cfg.get('k',10)),row.get('queries'),row.get('samples'),row.get('eligible_queries'),
                     row.get('recall',row.get('recall@10')),row.get('mrr',row.get('mrr@10')),row.get('ndcg',row.get('ndcg@10')),row.get('precision',row.get('precision@10')),
                     row.get('recall@1'),row.get('recall@5'),row.get('average_ms'),row.get('p50_ms'),row.get('p95_ms'),row.get('p99_ms'),
                     row.get('ram_mb'),row.get('embedding_ms'),row.get('search_ms'),row.get('indexing_ms'),row.get('model_loading_ms'),encode(row))
            conn.execute('INSERT INTO run_metrics VALUES('+','.join('?' for _ in metrics)+')',metrics)
        for row in value.get('detail',[]):
            metrics=(kind,key,row['index_id'],row['query_id'],row.get('repeat',1),row.get('objective','text'),row.get('query_type'),row.get('status'),
                     row.get('recall',row.get('recall@10')),row.get('mrr',row.get('mrr@10')),row.get('ndcg',row.get('ndcg@10')),row.get('precision',row.get('precision@10')),
                     row.get('total_ms'),row.get('embedding_ms'),row.get('search_ms'),row.get('ram_mb'),encode(row))
            conn.execute('INSERT INTO query_metrics VALUES('+','.join('?' for _ in metrics)+')',metrics)
        rankings=[]
        for sample in value.get('samples',[]):
            for row in sample['results']:
                rankings.append((kind,key,sample['index_id'],sample['query_id'],sample.get('repeat',1),sample.get('objective','text'),row['rank'],row.get('place_id'),row.get('image_id'),row.get('score'),encode(row)))
        conn.executemany('INSERT INTO search_results VALUES(?,?,?,?,?,?,?,?,?,?,?)',rankings)
