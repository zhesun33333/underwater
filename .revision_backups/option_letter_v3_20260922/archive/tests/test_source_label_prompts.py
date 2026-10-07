"""Regression checks for source-based L1 wording across generation and evaluation."""
import importlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
import wave

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from source_label_prompts import QA_PROMPT_VERSION, T1_TEMPLATES, get_t1_answer
from testsite.config import load_config
from testsite.core.parser import OutputParser

pulse = importlib.import_module("archive.pipeline_step2_qa")
ship = importlib.import_module("archive.pipeline_ship_step2_qa")

class SourceLabelTests(unittest.TestCase):
    def test_generation_and_evaluation_options_agree(self):
        cfg = load_config()
        self.assertEqual(cfg['qa_prompt_version'], QA_PROMPT_VERSION)
        self.assertEqual(cfg['prompts']['three_turn']['turn1'], T1_TEMPLATES[:3])
        parser = OutputParser(cfg)
        for q in T1_TEMPLATES:
            for answer, expected in [('A', 'active'), ('B', 'passive'),
                                      ('deliberately transmitted', 'active'),
                                      ('source-radiated noise', 'passive')]:
                self.assertEqual(parser.parse_turn1('sample', answer, prompt=q).L1, expected)

    def test_all_classes_generate_consistent_dialogues(self):
        parser = OutputParser(load_config())
        for module, category, classes in [
            (pulse, 'pulse', ['CW', 'LFM', 'HFM']),
            (pulse, 'communication', ['2FSK', '4FSK', 'BPSK', 'QPSK', 'OFDM']),
            (ship, 'radiated_noise', ['cargo', 'cruise', 'fishing', 'warship', 'underwater_target']),
        ]:
            expected = 'passive' if category == 'radiated_noise' else 'active'
            for l3 in classes:
                meta = {'signal_category': category, 'signal_type': l3}
                if expected == 'passive': meta['sub_type'] = l3
                labels = module.extract_labels(meta)
                for seed in range(40):
                    conversation = module.build_conversations(labels, meta, random.Random(seed))
                    self.assertEqual(len(conversation), 6)
                    text = json.dumps(conversation).lower()
                    for forbidden in ('passively received', 'passive listening', 'passive reception'):
                        self.assertNotIn(forbidden, text)
                    pred = parser.parse_turn1(l3, conversation[1]['value'], prompt=conversation[0]['value'])
                    self.assertEqual(pred.L1, expected)
                    pred2 = parser.parse(l3, conversation[3]['value'], prompt=conversation[2]['value'])
                    self.assertEqual(pred2.L1, expected)
                    self.assertEqual(pred2.L3, l3)

    def test_unknown_label_is_not_silently_passive(self):
        for invalid in ('unknown', '', 'typo', 'passively received'):
            with self.assertRaises(ValueError): get_t1_answer(invalid)

    def test_record_generation_preserves_identity_and_adds_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with wave.open(str(folder/'sample.wav'), 'wb') as wav:
                wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                wav.writeframes(b'\0\0' * 16)
            for module, category, l3 in [(pulse, 'pulse', 'LFM'), (ship, 'radiated_noise', 'cargo')]:
                metadata = {'id': 'sample_ch0', 'wav_path': 'sample.wav',
                            'signal_category': category, 'signal_type': l3, 'sub_type': l3}
                path = folder/'sample.json'
                path.write_text(json.dumps(metadata), encoding='utf-8')
                result = module.process_single_json(path, folder, random.Random(42))
                self.assertEqual(result['id'], 'sample_ch0')
                self.assertEqual(result['audio'], 'sample.wav')
                self.assertEqual(result['qa_prompt_version'], QA_PROMPT_VERSION)
                self.assertEqual(result['_meta']['L3'], l3)

if __name__ == '__main__':
    unittest.main()
