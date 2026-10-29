"""Fetch locked upstream baseline source checkouts; never run upstream code."""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'third_party'/'baselines.lock.json'
PATTERNS = ('*.py', '*.pyi', '*.pyx', '*.pxd', '*.c', '*.cpp', '*.h', '*.cu',
            '*.yaml', '*.yml', '*.toml', '*.lock', '*.cfg', '*.conf', '*.ini',
            '*.md', '*.rst', '*.txt', '*.sh', '*.json', '*.in',
            'LICENSE*', 'LICENCE*', 'COPYING*', 'NOTICE*', 'Makefile', '.gitignore')


def safe_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or any(p in ('..', '.', '.git') for p in path.parts)
            or any(c in value for c in '\\:\r\n\0*?[]!#') or value != path.as_posix()):
        raise ValueError('unsafe source path: '+value)
    return path


def load_lock(path=LOCK):
    lock = json.loads(Path(path).read_text(encoding='utf-8'))
    if lock['version'] != 1:
        raise ValueError('unsupported lockfile version')
    ids, names = set(), set()
    for source in lock['sources']:
        identifier = source['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in ids:
            raise ValueError('invalid or duplicate source id')
        ids.add(identifier)
        if not re.fullmatch(r'https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.git', source['url']):
            raise ValueError('source must use a public GitHub HTTPS repository URL')
        for field in ('commit', 'tree'):
            if not re.fullmatch('[0-9a-f]{40}', source[field]):
                raise ValueError('commit and tree must be full SHA-1 identifiers')
        for name, files in source['models'].items():
            if name in names or not files:
                raise ValueError('duplicate model or missing source paths')
            names.add(name)
            for value in files:
                safe_path(value)
    return lock


def select_sources(lock, names):
    known = {name for s in lock['sources'] for name in s['models']}
    unknown = set(names)-known
    if unknown:
        raise ValueError('unknown baseline(s): '+', '.join(sorted(unknown)))
    return [s for s in lock['sources'] if set(s['models']) & set(names)]


def git(args, cwd, stdin=None, timeout=240):
    env = os.environ.copy()
    # Public read-only fetches use no credential helper, user hooks or LFS smudge.
    env.update(GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never',
               GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
               GIT_LFS_SKIP_SMUDGE='1')
    for key in list(env):
        if key.startswith('GIT_CONFIG_KEY_') or key.startswith('GIT_CONFIG_VALUE_'):
            env.pop(key)
    for key in ('GIT_CONFIG_COUNT','GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE',
                'GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES'):
        env.pop(key, None)
    # File-backed capture avoids intermittent empty pipe reads on Windows.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        result = subprocess.run(['git', '-c', 'core.autocrlf=false',
                                 '-c', 'maintenance.auto=false', '-c', 'gc.auto=0',
                                 '-c', 'credential.helper=', *args], cwd=str(cwd), env=env,
                                input=stdin, stdout=out, stderr=err, timeout=timeout)
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(), err.read()
    if result.returncode:
        raise RuntimeError('git '+args[0]+' failed: '+stderr.decode('utf-8',errors='replace').strip())
    return stdout


def source_entries(folder):
    entries = git(['ls-tree','-r','-z','HEAD'], folder).split(b'\0')
    paths = {}
    for entry in entries:
        if not entry:
            continue
        meta, encoded = entry.split(b'\t',1)
        mode, kind, sha = meta.split()
        path = encoded.decode('utf-8')
        if kind != b'blob' or mode not in (b'100644', b'100755'):
            continue  # No submodules or symlinks are checked out.
        if any(fnmatch.fnmatchcase(PurePosixPath(path).name, p) for p in PATTERNS):
            safe_path(path)
            paths[path] = sha.decode('ascii')
    return paths


def selected_files(folder):
    return list(source_entries(folder))


def validate(folder, source):
    if folder.is_symlink() or not (folder/'.git').is_dir() or (folder/'.git').is_symlink():
        raise ValueError('expected a standalone checkout: '+str(folder))
    # Disable hooks even for inspection of existing fetches.
    head = git(['rev-parse','HEAD'],folder).decode().strip()
    if head != source['commit']:
        raise ValueError(f'checkout revision differs from lock: {source["id"]}; expected {source["commit"]!r}, got {head!r}')
    if git(['rev-parse','HEAD^{tree}'],folder).decode().strip() != source['tree']:
        raise ValueError('checkout tree differs from lock: '+source['id'])
    if git(['remote','get-url','origin'],folder).decode().strip() != source['url']:
        raise ValueError('checkout origin differs from lock: '+source['id'])
    # diff-index ignores untracked experiment outputs but detects tracked source edits.
    if git(['status','--porcelain','--untracked-files=no'],folder).strip():
        raise ValueError('tracked source edits detected; preserve this checkout: '+source['id'])
    paths = source_entries(folder)
    if not paths:
        raise ValueError('no selected source files in checkout')
    for path, expected in paths.items():
        actual = folder/path
        if not actual.is_file() or actual.is_symlink() or not actual.resolve().is_relative_to(folder.resolve()):
            raise ValueError('missing or unsafe source file: '+path)
        content = actual.read_bytes()
        observed = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if observed != expected:
            raise ValueError('source content differs from locked Git blob: '+path)
    for files in source['models'].values():
        for path in files:
            if path not in paths:
                raise ValueError('required model entry missing: '+path)
    return {'id':source['id'], 'models':list(source['models']), 'commit':source['commit'],
            'tree':source['tree'], 'source_files':len(paths), 'path':str(folder.resolve())}


def fetch(source, root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('download root cannot be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    target = root/source['id']
    if target.exists() or target.is_symlink():
        return validate(target, source)  # Never reset, update or overwrite an existing checkout.
    stage = Path(tempfile.mkdtemp(prefix='.'+source['id']+'-', dir=str(root)))
    try:
        git(['init','--template=','.'], stage)
        git(['config','core.hooksPath',str(stage/'.git'/'disabled-hooks')],stage)
        git(['remote','add','origin',source['url']],stage)
        git(['fetch','--depth=1','--filter=blob:none','--no-tags','origin',source['commit']],stage)
        # Set HEAD without materializing an unrestricted upstream working tree.
        git(['update-ref','HEAD',source['commit']],stage)
        if git(['rev-parse','HEAD^{tree}'],stage).decode().strip() != source['tree']:
            raise ValueError('fetched tree differs from lock')
        paths = selected_files(stage)
        rules = ''.join('/'+path+'\n' for path in paths).encode('utf-8')
        git(['sparse-checkout','set','--no-cone','--stdin'],stage,stdin=rules)
        git(['checkout','--detach',source['commit']],stage)
        validate(stage,source)
        # Windows scanners can briefly hold a freshly fetched directory open.
        for attempt in range(4):
            try:
                stage.rename(target)
                break
            except OSError as exc:
                if getattr(exc, 'winerror', None) not in (5,32) or attempt == 3:
                    raise
                time.sleep(.5*(attempt+1))
        git(['config','core.hooksPath',str(target/'.git'/'disabled-hooks')],target)
        return validate(target,source)
    except Exception as exc:
        raise RuntimeError(f'{source["id"]}: {exc}. Partial checkout retained at {stage}') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true', help='list locked models without network access')
    action.add_argument('--all', action='store_true', help='select all ten upstream models')
    action.add_argument('--models', nargs='+', help='select baseline names, e.g. DCRNN TSFlow')
    parser.add_argument('--check', action='store_true', help='verify existing checkouts without fetching')
    parser.add_argument('--root',type=Path,default=ROOT/'third_party'/'upstream')
    args = parser.parse_args(argv)
    lock = load_lock()
    if args.list:
        for s in lock['sources']:
            print(f'{", ".join(s["models"])} | {s["url"]} | {s["commit"]}')
        return 0
    try:
        sources = lock['sources'] if args.all else select_sources(lock,args.models)
    except ValueError as exc:
        parser.error(str(exc))
    failed = False
    for source in sources:
        try:
            report = validate(args.root/source['id'],source) if args.check else fetch(source,args.root)
            print(json.dumps(report), flush=True)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            failed = True
            print(str(exc),file=sys.stderr,flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
