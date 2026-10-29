"""Offline lockfile, checkout and non-destructive retrieval tests."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fetch_baselines', ROOT/'tools'/'fetch_baselines.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


class LockTests(unittest.TestCase):
    def test_all_ten_models_and_deduplication(self):
        lock = fetcher.load_lock()
        names = {name for source in lock['sources'] for name in source['models']}
        self.assertEqual(names, {'DCRNN','DGCRN','GWNET','STGCN','HimNet','STD-MAE',
                                 'TSFlow','UniST','STUM','FreqFlow'})
        self.assertEqual(len(lock['sources']),7)
        self.assertEqual(len(fetcher.select_sources(lock,['DCRNN','DGCRN','TSFlow'])),2)
        with self.assertRaises(ValueError):
            fetcher.select_sources(lock,['unknown'])

    def test_path_validation(self):
        for value in ('../escape','/root','a/../../b','C:/file','a\\b','a\nfile','a/.git/file','a/*'):
            with self.assertRaises(ValueError):
                fetcher.safe_path(value)
        self.assertEqual(str(fetcher.safe_path('models/model.py')),'models/model.py')

    def test_floating_revision_and_remote_rejected(self):
        for field,value in [('commit','main'),('tree','abcd'),('url','file:///tmp/repo')]:
            lock = copy.deepcopy(fetcher.load_lock())
            lock['sources'][0][field] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'lock.json'
                path.write_text(json.dumps(lock),encoding='utf-8')
                with self.assertRaises(ValueError):
                    fetcher.load_lock(path)


class CheckoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root/'fixture'
        self.repo.mkdir()
        fetcher.git(['init','--template=','.'],self.repo)
        for path,content in [('README.md',b'fixture\n'),('model.py',b'x = 1\n'),
                              ('weights.pt',b'not downloaded'),('data.npy',b'not downloaded')]:
            (self.repo/path).write_bytes(content)
        fetcher.git(['add','.'],self.repo)
        fetcher.git(['-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture'],self.repo)
        self.source = {'id':'Fixture','url':'https://github.com/example/fixture.git',
                       'commit':fetcher.git(['rev-parse','HEAD'],self.repo).decode().strip(),
                       'tree':fetcher.git(['rev-parse','HEAD^{tree}'],self.repo).decode().strip(),
                       'models':{'Fixture':['model.py','README.md']}}
        fetcher.git(['remote','add','origin',self.source['url']],self.repo)

    def test_exact_content_and_revision(self):
        self.assertEqual(fetcher.validate(self.repo,self.source)['source_files'],2)
        changed = dict(self.source, commit='0'*40)
        with self.assertRaises(ValueError):
            fetcher.validate(self.repo,changed)
        (self.repo/'model.py').write_text('x = 2\n',encoding='utf-8')
        fetcher.git(['update-index','--assume-unchanged','model.py'],self.repo)
        with self.assertRaises(ValueError):
            fetcher.validate(self.repo,self.source)

    def test_source_only_fetch_and_idempotent(self):
        original_git = fetcher.git
        def offline_git(args,cwd,**kwargs):
            if args[0] == 'fetch':
                args = ['fetch','--no-tags',str(self.repo),self.source['commit']]
            return original_git(args,cwd,**kwargs)
        root = self.root/'downloads'
        with patch.object(fetcher,'git',side_effect=offline_git):
            report = fetcher.fetch(self.source,root)
        target = root/'Fixture'
        self.assertEqual(report['source_files'],2)
        self.assertTrue((target/'model.py').is_file())
        self.assertFalse((target/'weights.pt').exists())
        self.assertFalse((target/'data.npy').exists())
        self.assertEqual(fetcher.fetch(self.source,root)['commit'],self.source['commit'])
        (target/'model.py').write_text('local edit\n',encoding='utf-8')
        with self.assertRaises(ValueError):
            fetcher.fetch(self.source,root)
        self.assertEqual((target/'model.py').read_text(),'local edit\n')

    def test_existing_non_repository_preserved(self):
        root = self.root/'downloads'
        target = root/'Fixture'
        target.mkdir(parents=True)
        marker = target/'important.txt'
        marker.write_text('preserve me',encoding='utf-8')
        with self.assertRaises(ValueError):
            fetcher.fetch(self.source,root)
        self.assertEqual(marker.read_text(),'preserve me')


if __name__ == '__main__':
    unittest.main()
