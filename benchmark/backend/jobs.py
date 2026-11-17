import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import data


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, kind):
        self.cancel = threading.Event()
        self.value = dict(id=uuid.uuid4().hex, kind=kind, created_at=data.now(), status='queued', current=0, total=0, logs=[])
        self.persist()

    def persist(self):
        data.save('jobs', self.value['id'], self.value)

    def check(self):
        if self.cancel.is_set():
            raise Cancelled('Đã hủy theo yêu cầu')

    def update(self, current, total, message):
        self.check()
        self.value.update(current=current, total=total)
        self.value['logs'].append(dict(at=data.now(), message=message))
        self.value['logs'] = self.value['logs'][-300:]
        self.persist()


POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='benchmark')
ACTIVE = {}


def submit(kind, function):
    job = Job(kind)
    ACTIVE[job.value['id']] = job

    def run():
        job.value['status'] = 'running'
        job.persist()
        try:
            job.check()
            job.value['result'] = function(job)
            job.check()
            job.value['status'] = 'completed'
        except Cancelled as exc:
            job.value.update(status='cancelled', error=str(exc))
        except Exception as exc:
            job.value.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        finally:
            job.value['finished_at'] = data.now()
            job.persist()
            ACTIVE.pop(job.value['id'], None)

    POOL.submit(run)
    return job.value.copy()


def recover():
    for group in ('image_indexes','image_runs','image_imports'):
        for value in data.listing(group):
            if value['status'] in ('building','running'):
                value.update(status='interrupted',error='Server restarted during image task; rerun required.')
                data.save(group,value['id'],value)
    for value in data.listing('jobs'):
        if value['status'] in ('queued', 'running'):
            value.update(status='interrupted', error='Server đã khởi động lại. Hãy chạy lại tác vụ.')
            data.save('jobs', value['id'], value)
    for index in data.listing('indexes'):
        if index['status'] == 'building':
            index.update(status='failed', error='Server bị ngắt khi indexing; index này không được dùng.')
            data.save('indexes', index['id'], index)
    for run in data.listing('runs'):
        if run['status'] == 'running':
            run.update(status='interrupted', error='Server đã khởi động lại; không phải run hoàn tất.')
            data.save('runs', run['id'], run)
