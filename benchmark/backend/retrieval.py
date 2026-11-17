import gc
import importlib.metadata
import os
import platform
import threading
import time
import unicodedata
import uuid

import httpx
import numpy as np
import psutil
from pydantic import BaseModel, Field, model_validator
from typing import Literal

from . import data

MODELS = {
    'M1': dict(model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', dimension=384, max_length=128, pooling='mean', query_prefix='', document_prefix=''),
    'M2': dict(model_id='intfloat/multilingual-e5-base', dimension=768, max_length=512, pooling='masked mean', query_prefix='query: ', document_prefix='passage: '),
    'M3': dict(model_id='intfloat/multilingual-e5-large', dimension=1024, max_length=512, pooling='masked mean', query_prefix='query: ', document_prefix='passage: '),
    'M4': dict(model_id='BAAI/bge-m3', dimension=1024, max_length=8192, pooling='CLS', query_prefix='', document_prefix=''),
}
ES_URL = os.getenv('ELASTICSEARCH_URL', 'http://127.0.0.1:9200').rstrip('/')
QDRANT_URL = os.getenv('QDRANT_URL', 'http://127.0.0.1:6333')


class IndexConfig(BaseModel):
    dataset_id: str
    method: Literal['BM25', 'M1', 'M2', 'M3', 'M4'] = 'BM25'
    fields: list[str] = Field(default_factory=lambda: ['name', 'category', 'description', 'tags'], min_length=1)
    k1: float = Field(default=1.2, ge=0, le=10)
    b: float = Field(default=0.75, ge=0, le=1)
    analyzer: Literal['vi_unicode', 'vi_folded'] = 'vi_unicode'
    revision: str = 'main'
    device: Literal['cpu', 'cuda'] = 'cpu'
    batch_size: int = Field(default=16, ge=1, le=128)
    normalize: bool = True
    max_length: int | None = Field(default=None, ge=1)

    @model_validator(mode='after')
    def check_fields(self):
        if set(self.fields) - set(data.FIELDS) or len(set(self.fields)) != len(self.fields):
            raise ValueError('Trường indexing không hợp lệ hoặc trùng')
        if self.method != 'BM25' and self.max_length and self.max_length > MODELS[self.method]['max_length']:
            raise ValueError('Max length vượt giới hạn checkpoint')
        return self


def resources():
    return {'ram_mb': psutil.Process().memory_info().rss / 2**20}


def environment():
    versions = {}
    for name in ('fastapi', 'numpy', 'pandas', 'qdrant-client', 'sentence-transformers', 'torch', 'transformers', 'huggingface-hub'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(platform=platform.platform(), python=platform.python_version(), cpu=platform.processor(),
                logical_cpus=psutil.cpu_count(), total_ram_mb=psutil.virtual_memory().total / 2**20, versions=versions)


def es(method, path, **kwargs):
    with httpx.Client(timeout=120) as client:
        res = client.request(method, ES_URL + '/' + path.lstrip('/'), **kwargs)
        res.raise_for_status()
        return res.json()


def qclient():
    from qdrant_client import QdrantClient
    # Local Qdrant storage is allowed for software tests, server URL is the default.
    path = os.getenv('QDRANT_LOCAL_PATH')
    return QdrantClient(path=path) if path else QdrantClient(url=QDRANT_URL, timeout=120)


def preprocess(value):
    return ' '.join(unicodedata.normalize('NFC', value).split())


class ModelManager:
    """One resident model, protected across requests, jobs and unload."""
    def __init__(self):
        self.lock = threading.RLock()
        self.model = None
        self.key = None
        self.info = None

    def unload(self):
        with self.lock:
            self.model = None
            self.key = None
            self.info = None
            gc.collect()
            import sys
            torch = sys.modules.get('torch')
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, config):
        with self.lock:
            spec = MODELS[config.method]
            key = (config.method, config.revision, config.device, config.max_length)
            if self.key == key:
                return 0.0
            self.unload()
            start = time.perf_counter()
            from huggingface_hub import HfApi, snapshot_download
            from sentence_transformers import SentenceTransformer
            # Resolve a moving ref once, and pin every subsequent load to this commit.
            patterns = ['config.json', 'config_sentence_transformers.json', 'modules.json',
                                                     'sentence_bert_config.json', 'tokenizer.json', 'tokenizer_config.json',
                                                     'special_tokens_map.json', 'added_tokens.json', 'vocab.txt',
                                                     'sentencepiece.bpe.model', 'spiece.model', 'model.safetensors',
                                                     'model-*.safetensors', 'model.safetensors.index.json',
                                                     '1_Pooling/config.json', '2_Normalize/config.json']
            remote = HfApi().model_info(spec['model_id'], revision=config.revision)
            if not any(f.rfilename.endswith('.safetensors') and '/' not in f.rfilename for f in remote.siblings):
                patterns += ['pytorch_model.bin', 'pytorch_model-*.bin', 'pytorch_model.bin.index.json']
            try:
                path = snapshot_download(spec['model_id'], revision=remote.sha, allow_patterns=patterns)
            except OSError as exc:
                if getattr(exc, 'winerror', None) != 1314:
                    raise
                # Windows accounts may lack symlink privilege, including nested
                # pooling files even when root files were downloaded successfully.
                revision = remote.sha
                path = snapshot_download(spec['model_id'], revision=revision, allow_patterns=patterns,
                                         local_dir=data.ROOT / 'text_model_cache' / config.method / revision,
                                         max_workers=1)
            resolved = os.path.basename(path)
            model = SentenceTransformer(path, device=config.device, trust_remote_code=False)
            model.max_seq_length = config.max_length or spec['max_length']
            if model.get_sentence_embedding_dimension() != spec['dimension']:
                raise ValueError('Dimension checkpoint khác model registry')
            from sentence_transformers.models import Normalize
            # Some checkpoints include Normalize internally. Remove only that final
            # operation so the experiment normalization toggle is effective.
            import torch
            model._modules = {k: v for k, v in model._modules.items() if not isinstance(v, Normalize)}
            self.model, self.key = model, key
            self.info = dict(spec, revision=resolved, device=config.device, max_length=model.max_seq_length,
                             architecture=str(model), gpu=torch.cuda.get_device_name() if config.device == 'cuda' else None)
            return (time.perf_counter() - start) * 1000

    def encode(self, texts, config, query=False):
        prefix = MODELS[config.method]['query_prefix' if query else 'document_prefix']
        values = [prefix + preprocess(s) for s in texts]
        return np.asarray(self.model.encode(values, batch_size=config.batch_size, normalize_embeddings=config.normalize,
                                             convert_to_numpy=True, show_progress_bar=False), dtype=np.float32)


manager = ModelManager()


def build_index(config, job):
    dataset = data.get('datasets', config.dataset_id)
    if dataset['kind'] != 'places':
        raise ValueError('Cần chọn dataset địa điểm')
    rows = dataset['records']
    started = time.perf_counter()
    identity = 'lab-' + uuid.uuid4().hex
    meta = dict(id=identity, created_at=data.now(), config=config.model_dump(), dataset_sha256=dataset['sha256'],
                count=len(rows), preprocessing='Unicode NFC + collapse whitespace; fields joined by .', status='building')
    data.save('indexes', identity, meta)
    client = None
    created_collections = []
    try:
        job.update(0, len(rows), 'Khởi tạo index riêng')
        if config.method == 'BM25':
            filters = ['lowercase'] + (['asciifolding'] if config.analyzer == 'vi_folded' else [])
            analysis = {'analyzer': {'lab_vi': {'type': 'custom', 'tokenizer': 'standard', 'filter': filters}}}
            settings = {'analysis': analysis, 'similarity': {'lab_bm25': {'type': 'BM25', 'k1': config.k1, 'b': config.b}}}
            mappings = {'properties': {f: {'type': 'text', 'analyzer': 'lab_vi', 'search_analyzer': 'lab_vi', 'similarity': 'lab_bm25'} for f in config.fields}}
            es('PUT', identity, json={'settings': settings, 'mappings': mappings})
            meta.update(analyzer=analysis, model_loading_ms=0, embedding_ms=0)
            import json
            for offset in range(0, len(rows), 200):
                job.check()
                lines = []
                for row in rows[offset:offset + 200]:
                    doc = {f: preprocess('; '.join(row[f]) if isinstance(row[f], list) else row[f]) for f in config.fields}
                    lines += [json.dumps({'index': {'_index': identity, '_id': row['place_id']}}), json.dumps(doc, ensure_ascii=False)]
                result = es('POST', '_bulk', content=('\n'.join(lines) + '\n').encode(), headers={'Content-Type': 'application/x-ndjson'})
                if result.get('errors'):
                    raise RuntimeError('Elasticsearch bulk lỗi: ' + str(result['items'][:3]))
                job.update(min(offset + 200, len(rows)), len(rows), 'Đang index BM25')
            es('POST', identity + '/_refresh')
            meta['indexed_count']=es('GET',identity+'/_count')['count']
            if meta['indexed_count']!=len(rows):
                raise ValueError('Số tài liệu Elasticsearch không khớp dataset')
            stats = es('GET', identity + '/_stats/store')
            meta['index_size_bytes'] = stats['_all']['primaries']['store']['size_in_bytes']
            meta['server_version'] = es('GET', '')['version']['number']
        else:
            from qdrant_client import models as qm
            with manager.lock:
                job.check()
                job.update(0, len(rows), 'Đang tải / nạp checkpoint embedding; lần đầu phụ thuộc tốc độ mạng')
                meta['model_loading_ms'] = manager.load(config)
                meta['model'] = manager.info.copy()
                # Persist the immutable revision in the searchable configuration.
                config = config.model_copy(update={'revision': manager.info['revision']})
                manager.key = (config.method, config.revision, config.device, config.max_length)
                meta['config'] = config.model_dump()
                client = qclient()
                meta['collections'] = {}
                for metric, distance in [('Cosine', qm.Distance.COSINE), ('Dot', qm.Distance.DOT), ('Euclid', qm.Distance.EUCLID)]:
                    collection = identity + '-' + metric.lower()
                    client.create_collection(collection, vectors_config=qm.VectorParams(size=MODELS[config.method]['dimension'], distance=distance))
                    created_collections.append(collection)
                    meta['collections'][metric] = collection
                vectors = []
                embed_ms = 0
                for offset in range(0, len(rows), config.batch_size):
                    job.check()
                    batch = rows[offset:offset + config.batch_size]
                    start = time.perf_counter()
                    values = manager.encode([data.text(r, config.fields) for r in batch], config)
                    if values.shape != (len(batch), MODELS[config.method]['dimension']) or not np.isfinite(values).all():
                        raise ValueError('Embedding không hợp lệ: sai kích thước hoặc chứa NaN/Infinity')
                    embed_ms += (time.perf_counter() - start) * 1000
                    vectors.append(values)
                    points = [qm.PointStruct(id=offset + i, vector=v.tolist(), payload={'place_id': row['place_id']}) for i, (row, v) in enumerate(zip(batch, values))]
                    for collection in created_collections:
                        client.upsert(collection, points=points, wait=True)
                    job.update(min(offset + config.batch_size, len(rows)), len(rows), 'Embedding và index 3 metric từ cùng vector')
                meta['collection_counts'] = {metric: client.count(collection, exact=True).count for metric, collection in meta['collections'].items()}
                if any(count != len(rows) for count in meta['collection_counts'].values()):
                    raise ValueError('Số vector Qdrant không khớp số địa điểm')
                meta['indexed_count'] = len(rows)
                vector_path = data.ROOT / 'vectors'
                vector_path.mkdir(exist_ok=True, parents=True)
                file = vector_path / (identity + '.npy')
                np.save(file, np.vstack(vectors))
                meta.update(embedding_ms=embed_ms, vector_artifact_bytes=file.stat().st_size,
                            index_size_bytes=None, index_size_note='Qdrant không cung cấp dung lượng collection qua API chuẩn; vector_artifact_bytes là tệp embedding, không phải index size.',
                            server_version=client.info().version)
        job.check()
        meta.update(status='ready', indexing_ms=(time.perf_counter() - started) * 1000, resources=resources())
        data.save('indexes', identity, meta)
        return meta
    except Exception as exc:
        meta.update(status='failed', error=str(exc))
        data.save('indexes', identity, meta)
        # Never expose a partially indexed collection as ready.
        if client:
            for collection in created_collections:
                client.delete_collection(collection)
        elif config.method == 'BM25':
            try:
                es('DELETE', identity)
            except Exception:
                pass
        raise
    finally:
        if client:
            client.close()


def normalize_metric(metric):
    if metric is None or not str(metric).strip():
        return 'Cosine'
    names={'cosine':'Cosine','dot':'Dot','euclid':'Euclid','euclidean':'Euclid'}
    value=names.get(str(metric).strip().casefold())
    if value is None:
        raise ValueError('Similarity của Dense Search phải là Cosine, Dot hoặc Euclid')
    return value


def search(index_id, query, k=10, metric='Cosine', exact=True):
    if not isinstance(query,str) or not query.strip():
        raise ValueError('Hãy nhập nội dung truy vấn, không chỉ khoảng trắng')
    if isinstance(k,bool) or not isinstance(k,int) or not 1 <= k <= 1000:
        raise ValueError('Top-K phải là số nguyên từ 1 đến 1000')
    index = data.get('indexes', index_id)
    if index['status'] != 'ready':
        raise ValueError('Index chưa sẵn sàng')
    config = IndexConfig(**index['config'])
    if config.method!='BM25':
        metric=normalize_metric(metric)
    rows = data.get('datasets', config.dataset_id)['records']
    lookup = {r['place_id']: r for r in rows}
    total = time.perf_counter()
    loading_ms = embedding_ms = 0
    if config.method == 'BM25':
        start = time.perf_counter()
        response = es('POST', index_id + '/_search', json={'size': k, 'query': {'multi_match': {'query': preprocess(query), 'fields': config.fields, 'type': 'most_fields'}}})
        search_ms = (time.perf_counter() - start) * 1000
        hits = [(h['_id'], h['_score']) for h in response['hits']['hits']]
        score_type = 'BM25'
    else:
        from qdrant_client import models as qm
        with manager.lock:
            loading_ms = manager.load(config)
            start = time.perf_counter()
            vector = manager.encode([query], config, query=True)[0]
            embedding_ms = (time.perf_counter() - start) * 1000
            client = qclient()
            try:
                start = time.perf_counter()
                response = client.query_points(collection_name=index['collections'][metric], query=vector.tolist(), limit=k, search_params=qm.SearchParams(exact=exact), with_payload=True)
                search_ms = (time.perf_counter() - start) * 1000
                hits = [(p.payload['place_id'], p.score) for p in response.points]
            finally:
                client.close()
        score_type = {'Cosine': 'Cosine similarity', 'Dot': 'Dot product', 'Euclid': 'Euclidean distance (nhỏ hơn tốt hơn)'}[metric]
    from . import place_media
    metadata_start=time.perf_counter()
    results=place_media.attach(config.dataset_id,[dict(lookup[pid],rank=i,score=float(score)) for i,(pid,score) in enumerate(hits,1)])
    image_metadata_ms=(time.perf_counter()-metadata_start)*1000
    return dict(index_id=index_id, method=config.method, query=query, k=k, metric=metric if config.method != 'BM25' else None, exact=exact,
                score_type=score_type, results=results,
                timing=dict(total_ms=(time.perf_counter() - total) * 1000, model_loading_ms=loading_ms, embedding_ms=embedding_ms, search_ms=search_ms,
                            phase='cold' if loading_ms else 'warm',image_metadata_ms=image_metadata_ms), resources=resources())
