"""Bounded metadata updates consume a charged reserve, without disk rescans.

The pool is charged against the project quota once. Under its storage lock, a
replacement first releases enough reserve for the entire temporary file, then
rebalances the remaining reserve. Metadata plus reserve never grows. No Torch.
"""
import errno
import json
import os
from pathlib import Path
from paisho_ppo import budget

POOL_BYTES=1024*1024
MAX_CONTENT_BYTES=512*1024
MARGIN_BYTES=16384


def charge(path):
    try:
        info=path.stat()
        return max(info.st_size,info.st_blocks*512)
    except FileNotFoundError:return 0


def replace(path,content):
    path=Path(path)
    if len(content)>MAX_CONTENT_BYTES:return False
    reserve=path.with_name('.'+path.name+'.reserve')
    temporary=path.with_suffix(path.suffix+'.tmp')
    try:
        with budget.project_write_guard(enforce=False):
            temporary.unlink(missing_ok=True)
            if not reserve.exists():
                budget.check(reserve=POOL_BYTES+budget.METADATA_RESERVE)
                with reserve.open('wb') as stream:
                    stream.truncate(POOL_BYTES);stream.flush();os.fsync(stream.fileno())
            pool=charge(reserve)+charge(path)
            required=((len(content)+4095)//4096)*4096+MARGIN_BYTES
            if pool-charge(path)<required:return False
            with reserve.open('r+b') as stream:
                stream.truncate(pool-charge(path)-required)
                stream.flush();os.fsync(stream.fileno())
                if pool-charge(path)-charge(reserve)<required:
                    stream.truncate(pool-charge(path));return False
                try:
                    with temporary.open('wb') as target:
                        target.write(content);target.flush();os.fsync(target.fileno())
                    os.replace(temporary,path)
                    fd=os.open(path.parent,os.O_RDONLY)
                    try:os.fsync(fd)
                    finally:os.close(fd)
                finally:
                    temporary.unlink(missing_ok=True)
                    stream.truncate(max(0,pool-charge(path)))
                    stream.flush();os.fsync(stream.fileno())
        return True
    except budget.SpaceUnavailable:return False
    except OSError as error:
        if error.errno not in (errno.ENOSPC,errno.EDQUOT):raise
        return False


def write(path,value):
    return replace(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
