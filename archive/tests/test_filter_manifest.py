import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from archive import filter_test_set


class FilterManifestTests(unittest.TestCase):
    def test_invalid_manifest_never_overwrites_output(self):
        row = {'id': 'sample', 'audio': 'sample.wav'}
        cases = [json.dumps(row)+'\n'+json.dumps(row),
                 json.dumps(row)+'\nBAD JSON', json.dumps({'id': 'missing_audio'}),
                 '', '[]']
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source, output = folder/'input.jsonl', folder/'output.jsonl'
            output.write_text('keep')
            for text in cases:
                source.write_text(text)
                with patch('sys.argv', ['filter', '--input', str(source), '--output', str(output)]), \
                     contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    filter_test_set.main()
                self.assertEqual(output.read_text(), 'keep')
