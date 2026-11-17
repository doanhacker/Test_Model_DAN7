"""Immutable local snapshots. Uploaded originals are never modified."""
import hashlib
import io
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from . import database

ROOT = Path(os.getenv('BENCHMARK_DATA_DIR', Path(__file__).resolve().parents[1] / 'runtime'))
DEFAULT_ROOT = ROOT
LOCK = threading.RLock()
SCHEMAS = {
    'places': ['place_id', 'name', 'category', 'city', 'address', 'description', 'tags', 'amenities'],
    'queries': ['query_id', 'query_text', 'query_type', 'split'],
    'qrels': ['query_id', 'place_id', 'relevance', 'reason', 'review_status'],
}
REQUIRED = {'places': ['place_id', 'name'], 'queries': ['query_id', 'query_text', 'query_type', 'split'], 'qrels': ['query_id','place_id','relevance']}
FIELDS = SCHEMAS['places'][1:]


def ram_only(value):
    """Hide retired memory metrics in historical API/export results without editing snapshots."""
    if isinstance(value, dict):
        return {k: ram_only(v) for k, v in value.items() if not k.lower().startswith('vram')}
    if isinstance(value, list):
        return [ram_only(v) for v in value]
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def db_path_override():
    # PostgreSQL test roots use isolated schemas, never the production tables.
    return os.getenv('DATABASE_URL')


def database_info():
    return database.info(ROOT,db_path_override())


def save(group, key, value):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', key):
        raise ValueError('ID không hợp lệ')
    with LOCK:
        database.save(ROOT,group,key,value,db_path_override())


def get(group, key):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', key):
        raise ValueError('ID không hợp lệ')
    with LOCK:
        return database.get(ROOT,group,key,db_path_override())


def listing(group):
    with LOCK:
        return database.listing(ROOT,group,db_path_override())


def parse(raw, filename):
    suffix = Path(filename).suffix.lower()
    if suffix == '.csv':
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    elif suffix == '.xlsx':
        frame = pd.read_excel(io.BytesIO(raw), dtype=str, keep_default_na=False)
    elif suffix == '.json':
        value = json.loads(raw.decode('utf-8-sig'))
        if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
            raise ValueError('JSON phải là mảng object')
        frame = pd.DataFrame(value).fillna('')
    else:
        raise ValueError('Chỉ hỗ trợ CSV, XLSX và JSON')
    if frame.columns.duplicated().any():
        raise ValueError('Tên cột bị trùng')
    return frame


def validate(kind, frame, mapping=None, dataset=None, queries=None):
    if kind not in SCHEMAS:
        raise ValueError('Loại dữ liệu không hợp lệ')
    mapping = mapping or {}
    # Mapping direction: canonical field -> uploaded column.
    renamed = {source: target for target, source in mapping.items() if source}
    frame = frame.rename(columns=renamed)
    if frame.columns.duplicated().any():
        raise ValueError('Mapping tạo tên cột trùng')
    missing = set(REQUIRED[kind]) - set(frame.columns)
    if missing:
        raise ValueError('Thiếu cột: ' + ', '.join(sorted(missing)))
    frame = frame.copy()
    for field in SCHEMAS[kind]:
        if field not in frame:
            frame[field] = ''
    records = frame[SCHEMAS[kind]].to_dict('records')
    if not records and kind != 'qrels':
        raise ValueError('Tệp dữ liệu rỗng')
    for i, row in enumerate(records, 2):
        for field in SCHEMAS[kind]:
            val = row[field]
            if field in ('tags', 'amenities'):
                if isinstance(val, str):
                    val = json.loads(val) if val.strip().startswith('[') else [v.strip() for v in val.split(';') if v.strip()]
                if not isinstance(val, list) or any(not isinstance(v, str) for v in val):
                    raise ValueError(f'Dòng {i}: {field} phải là mảng chuỗi hoặc phân cách bằng ;')
                row[field] = val
            else:
                row[field] = '' if pd.isna(val) else str(val).strip()
        if kind == 'qrels' and not row['review_status']:
            row['review_status']='pending'
        if any(row[c] == '' for c in REQUIRED[kind] if not (kind=='qrels' and c=='relevance')):
            raise ValueError(f'Dòng {i}: thiếu giá trị bắt buộc')
        if kind == 'queries' and row['split'] not in ('train', 'validation', 'test'):
            raise ValueError(f'Dòng {i}: split phải là train/validation/test')
        if kind == 'qrels':
            if row['review_status'] not in ('pending','proposed','confirmed'):
                raise ValueError('review_status phải là pending/proposed/confirmed')
            if row['relevance'] in ('','None') and row['review_status']!='confirmed':
                row['relevance']=None
            else:
                rel = float(row['relevance'])
                if not rel.is_integer() or not 0 <= rel <= 3:
                    raise ValueError(f'Dòng {i}: relevance phải là số nguyên 0–3')
                row['relevance'] = int(rel)
    keys = ['query_id', 'place_id'] if kind == 'qrels' else ['place_id' if kind == 'places' else 'query_id']
    ids = [tuple(r[c] for c in keys) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Phát hiện ID trùng: ' + ', '.join(keys))
    if kind == 'qrels':
        if not dataset or not queries:
            raise ValueError('Ground Truth phải liên kết dataset và query set')
        pids = {r['place_id'] for r in dataset['records']}
        qids = {r['query_id'] for r in queries['records']}
        if any(r['place_id'] not in pids or r['query_id'] not in qids for r in records):
            raise ValueError('Ground Truth chứa ID không tồn tại trong phiên bản đã chọn')
    return records


def import_records(kind, records, name, dataset_id=None, queries_id=None, original=None, mapping=None):
    ds = get('datasets', dataset_id) if dataset_id else None
    qs = get('datasets', queries_id) if queries_id else None
    if ds and ds['kind'] != 'places' or qs and qs['kind'] != 'queries':
        raise ValueError('Sai loại phiên bản tham chiếu')
    clean = validate(kind, pd.DataFrame(records) if records else pd.DataFrame(columns=SCHEMAS[kind]), mapping, ds, qs)
    value = {'kind': kind, 'records': clean, 'dataset_id': dataset_id, 'queries_id': queries_id}
    key = digest(value)[:24]
    try:
        return get('datasets', key)
    except FileNotFoundError:
        pass
    value.update(id=key, name=name, created_at=now(), count=len(clean), sha256=digest(value), mapping=mapping or {},
                 missing={f: sum(r[f] == '' or r[f] == [] for r in clean) for f in SCHEMAS[kind]})
    if original is not None:
        folder = ROOT / 'originals'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (key + '.bin')).write_bytes(original)
    save('datasets', key, value)
    return value


def text(row, fields):
    return ' . '.join('; '.join(row[f]) if isinstance(row[f], list) else str(row[f]) for f in fields)
