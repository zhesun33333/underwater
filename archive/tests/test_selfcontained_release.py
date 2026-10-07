"""Exercise serialized Step 2 output in a detached testsite-only release."""
import contextlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

import yaml
from archive import pipeline_step2_qa as pulse, pipeline_ship_step2_qa as ship
from archive import filter_test_set
from testsite.core.dataset_record import LEAF_PARENTS, resolve_source_audio
from testsite.core.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]


class SelfContainedReleaseTests(unittest.TestCase):
    def test_both_step2_cli_outputs_work_without_original_metadata_or_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            processed, qa = base / 'processed', base / 'qa'
            for leaf, (l1, l2) in LEAF_PARENTS.items():
                sid = f'{leaf}_000001_ch0'
                directory = (processed / 'PulseCom/jsonc' / leaf if l1 == 'active'
                             else processed / 'Ship' / leaf / 'json')
                directory.mkdir(parents=True)
                wav = processed / 'audio' / f'{sid}.wav'
                wav.parent.mkdir(exist_ok=True)
                with wave.open(str(wav), 'wb') as stream:
                    stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                    stream.writeframes(b'\x01\x00' * 160)
                meta = {
                    'id': sid, 'wav_path': wav.relative_to(processed).as_posix(),
                    'signal_category': l2 if l1 == 'active' else 'radiated_noise',
                    'signal_type': leaf, 'signal_params': {'sweep_direction': 'up'},
                    'bellhop_output': {'tl_db': -25.0} if l1 == 'active' else {'snr_db': 12.5},
                    'bellhop_env': {'ssp': {'depths_m': [0, 10, 20], 'sound_speeds_mps': [1500, 1502, 1506]}},
                }
                if l1 == 'passive':
                    meta['sub_type'] = leaf
                (directory / f'{sid}.jsonc').write_text('// source comment\n' + json.dumps(meta), encoding='utf-8')
            common = {'paths': {'processed_audio': str(processed), 'qa_output': str(qa)},
                      'limits': {'random_seed': 42}}
            configs = [
                (pulse, dict(common, datasets={'pulsecom': {'name': 'PulseCom', 'json_dir': 'jsonc', 'json_ext': '.jsonc'}})),
                (ship, dict(common, dataset={'name': 'Ship', 'classes': [k for k, v in LEAF_PARENTS.items() if v[0] == 'passive'],
                                            'json_subdir': 'json', 'json_ext': '.jsonc'})),
            ]
            for module, config in configs:
                cp = base / 'config.yaml'
                cp.write_text(yaml.safe_dump(config), encoding='utf-8')
                with patch('sys.argv', ['step2', '--config', str(cp)]), contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            records = [json.loads(line) for p in qa.glob('*.jsonl') for line in p.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(records), 13)
            for record in records:
                self.assertEqual(record['_meta']['source_id'], record['id'].removesuffix('_ch0'))
                self.assertAlmostEqual(record['_meta']['ssp_complexity'], 0.01)
                self.assertNotIn('\\', record['audio'])
                self.assertEqual(len(record['conversations']), 6)
            release = base / 'release'
            release.mkdir()
            shutil.copytree(ROOT / 'testsite', release / 'testsite', ignore=shutil.ignore_patterns('__pycache__', '*.tmp.*'))
            shutil.copytree(processed / 'audio', release / 'audio')
            manifest = release / 'records.jsonl'
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
            # Evaluation sees only the release directory. Original JSONC files
            # are not copied; only the two delivered components are present.
            env = dict(os.environ)
            env.pop('PYTHONPATH', None)
            result = subprocess.run([sys.executable, '-B', '-m', 'testsite.scripts.run_eval',
                                     '--data', 'records.jsonl', '--audio-root', '.', '--backend', 'mock',
                                     '--output-dir', 'results'], cwd=release, env=env,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outputs = [json.loads(s) for s in (release/'results/predictions_shard_000.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(outputs), 13)
            self.assertEqual({r['gt']['L3'] for r in outputs}, set(LEAF_PARENTS))
            # Subset selection also consumes the embedded statistics.
            selected = release / 'selected.jsonl'
            with patch.object(filter_test_set, 'find_meta', side_effect=AssertionError('must not recover original metadata')), patch(
                'sys.argv', ['filter', '--input', str(manifest), '--output', str(selected), '--n-per-class', '1']
            ), contextlib.redirect_stdout(io.StringIO()):
                filter_test_set.main()
            self.assertEqual(len(DataLoader({}).load(str(selected), str(release))), 13)

    def test_missing_labels_do_not_fall_back_to_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = {'id': 'sample', 'audio': 'sample.wav',
                      'conversations': [{'from': 'human', 'value': 'question'}] * 3}
            with self.assertRaisesRegex(ValueError, 'embedded _gt'):
                DataLoader({})._build_sample(record, Path(tmp))

    def test_missing_or_ambiguous_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for module, category, leaf in [(pulse, 'pulse', 'CW'), (ship, 'radiated_noise', 'cargo')]:
                meta = {'id': 'sample', 'signal_category': category, 'signal_type': leaf}
                p = root / 'sample.json'
                p.write_text(json.dumps(meta), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'wav_path'):
                    module.process_single_json(p, root, random.Random(42))
            for name in ['a', 'b']:
                (root / name).mkdir()
                (root / name / 'sample.wav').write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'found 2'):
                resolve_source_audio({'id': 'sample', 'wav_path': 'missing.wav'}, root)


if __name__ == '__main__':
    unittest.main()
