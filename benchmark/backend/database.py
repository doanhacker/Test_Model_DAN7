"""PostgreSQL server persistence. SQLite is read only once as a migration source."""
import atexit
import hashlib
import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
from psycopg import sql
from .storage_schema import SCHEMA,TABLES,encode,put_document,project

LOCK=threading.RLock()
READY=set()
TEST_SCHEMAS={}


class Record(dict):
    def __getitem__(self,key):
        return list(self.values())[key] if isinstance(key,int) else super().__getitem__(key)


def row_factory(cursor):
    columns=[column.name for column in cursor.description] if cursor.description else []
    return lambda values:Record(zip(columns,values))


def connection_url(explicit=None):
    value=explicit or os.getenv('DATABASE_URL')
    if not value or not value.startswith(('postgresql://','postgres://')):
        raise ValueError('DATABASE_URL phải là kết nối PostgreSQL; không dùng file SQLite')
    return value


def schema_for(root):
    production=Path(os.getenv('BENCHMARK_DATA_DIR',Path(__file__).resolve().parents[1]/'runtime')).resolve()
    root=Path(root).resolve()
    if root==production:
        schema=os.getenv('POSTGRES_SCHEMA','public')
    else:
        if not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise ValueError('ROOT khác production phải nằm trong thư mục test tạm')
        schema='test_'+hashlib.sha256(str(root).encode()).hexdigest()[:20]
        TEST_SCHEMAS[schema]=root
    if not re.fullmatch('[a-z][a-z0-9_]{0,62}',schema):
        raise ValueError('POSTGRES_SCHEMA không hợp lệ')
    return schema


class Connection:
    def __init__(self,root,explicit=None):
        self.schema=schema_for(root)
        self.raw=psycopg.connect(connection_url(explicit),connect_timeout=10,row_factory=row_factory,
                                 options='-c search_path='+self.schema)
    def execute(self,statement,params=None):
        return self.raw.execute(statement.replace('?','%s'),params)
    def executemany(self,statement,rows):
        with self.raw.cursor() as cursor:
            cursor.executemany(statement.replace('?','%s'),rows)
    def executescript(self,script):
        for statement in script.split(';'):
            if statement.strip():
                self.execute(statement)
    def close(self):
        self.raw.close()
    def __enter__(self):
        return self
    def __exit__(self,typ,value,traceback):
        self.raw.rollback() if typ else self.raw.commit()


def legacy_documents(root):
    """No runtime fallback: the old database remains an untouched migration backup."""
    source=Path(root)/'benchmark.sqlite3'
    if source.exists():
        import sqlite3
        conn=sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            rows=conn.execute('SELECT group_name,record_id,payload FROM documents').fetchall()
        finally:
            conn.close()
        return [(g,key,json.loads(payload)) for g,key,payload in rows],dict(source='sqlite_legacy',file=str(source))
    records=[]
    for source in sorted(Path(root).glob('*/*.json')):
        records.append((source.parent.name,source.stem,json.loads(source.read_text(encoding='utf-8'))))
    return records,dict(source='json_legacy')


def initialize(root,explicit=None):
    identity=(connection_url(explicit),schema_for(root))
    with LOCK:
        conn=Connection(root,explicit)
        try:
            with conn:
                conn.execute('SELECT pg_advisory_xact_lock(hashtext(?))',('benchmark-init-'+conn.schema,))
                conn.raw.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(conn.schema)))
                conn.executescript(SCHEMA)
                if not conn.execute("SELECT 1 FROM schema_meta WHERE key='server_storage_initialized'").fetchone():
                    records,origin=legacy_documents(root)
                    for group,key,value in records:
                        put_document(conn,group,key,value)
                    marker=dict(**origin,documents=len(records),at=datetime.now(timezone.utc).isoformat())
                    conn.execute('INSERT INTO schema_meta VALUES(?,?)',('server_storage_initialized',encode(marker)))
                    conn.execute('INSERT INTO schema_meta VALUES(?,?)',('schema_version','2'))
                conn.execute("INSERT INTO schema_meta VALUES('schema_version','3') ON CONFLICT(key) DO UPDATE SET value=excluded.value")
        finally:
            conn.close()
        READY.add(identity)
    return public_address(explicit)


@contextmanager
def session(root,explicit=None):
    identity=(connection_url(explicit),schema_for(root))
    if identity not in READY:
        initialize(root,explicit)
    conn=Connection(root,explicit)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def save(root,group,key,value,explicit=None):
    with LOCK,session(root,explicit) as conn:
        put_document(conn,group,key,value)


def get(root,group,key,explicit=None):
    with session(root,explicit) as conn:
        row=conn.execute('SELECT payload FROM documents WHERE group_name=? AND record_id=?',(group,key)).fetchone()
    if row is None:
        raise FileNotFoundError(f'{group}/{key}')
    return json.loads(row['payload'])


def listing(root,group,explicit=None):
    with session(root,explicit) as conn:
        rows=conn.execute('SELECT payload FROM documents WHERE group_name=? ORDER BY created_at DESC,record_id',(group,)).fetchall()
    return [json.loads(r['payload']) for r in rows]


def public_address(explicit=None):
    parsed=urlsplit(connection_url(explicit))
    return f'postgresql://{parsed.username}@{parsed.hostname}:{parsed.port or 5432}{parsed.path}'


def info(root,explicit=None):
    with session(root,explicit) as conn:
        counts={table:conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in TABLES}
        marker=conn.execute("SELECT value FROM schema_meta WHERE key='server_storage_initialized'").fetchone()
        size=conn.execute('SELECT pg_database_size(current_database())').fetchone()[0]
        version=conn.execute('SHOW server_version').fetchone()[0]
    parsed=urlsplit(connection_url(explicit))
    return dict(engine='PostgreSQL',path=public_address(explicit),host=parsed.hostname,port=parsed.port or 5432,
                name=parsed.path.lstrip('/'),schema=schema_for(root),server_version=version,size_bytes=size,counts=counts,
                legacy_migration=json.loads(marker[0]) if marker else None)


def backup(root,target,explicit=None):
    """Portable psql SQL dump from a repeatable-read PostgreSQL transaction."""
    with session(root,explicit) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        with open(target,'wb') as output:
            output.write(('-- PostgreSQL benchmark backup\nBEGIN;\n'+SCHEMA+'\n').encode('utf-8'))
            for table in ('schema_meta',)+TABLES:
                names=conn.execute('SELECT column_name FROM information_schema.columns WHERE table_schema=? AND table_name=? ORDER BY ordinal_position',(conn.schema,table)).fetchall()
                columns=sql.SQL(',').join(sql.Identifier(r[0]) for r in names)
                command=sql.SQL('COPY {} ({}) FROM stdin;\n').format(sql.Identifier(table),columns)
                output.write(command.as_string(conn.raw).encode('utf-8'))
                with conn.raw.cursor() as cursor:
                    with cursor.copy(sql.SQL('COPY {}.{} ({}) TO STDOUT').format(sql.Identifier(conn.schema),sql.Identifier(table),columns)) as copied:
                        for block in copied:
                            output.write(bytes(block))
                output.write(b'\\.\n')
            output.write(b'COMMIT;\n')


def cleanup_test_schemas():
    if not TEST_SCHEMAS:
        return
    try:
        with psycopg.connect(connection_url(),autocommit=True,connect_timeout=5) as conn:
            for schema,root in TEST_SCHEMAS.items():
                if re.fullmatch('test_[0-9a-f]{20}',schema) and not root.exists():
                    conn.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
    except psycopg.Error:
        pass


atexit.register(cleanup_test_schemas)
