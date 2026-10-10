"""Source identity, selection, structural scaling and figure-C cache contracts."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from visualization.channel_data import (load_channels, normalize_peak, prepare_channels,
                                        read_source_members, select_channel_pairs, validate_source_pair)


CLASSES = ('LFM', '2FSK', 'cruise')


def fixtures():
    records, source_metadata, received_metadata, waves = [], {}, {}, {}
    for leaf, family in zip(CLASSES, ('pulse', 'communication', 'ship_noise')):
        for index in range(2):
            sid = f'{leaf}_{index}'
            params = {'center_freq_hz': 500, 'carrier_freq_hz': 500, 'bandwidth_hz': 200,
                      'symbol_rate_baud': 100}
            source_metadata[sid] = {'id': sid, 'signal_type': leaf, 'sub_type': leaf,
                                     'signal_params': params, 'wav_path': sid+'.wav', 'audio_duration_s': 30}
            buffer = io.BytesIO()
            sf.write(buffer, .4*np.sin(2*np.pi*500*np.arange(1600)/16000), 16000, format='WAV', subtype='PCM_32')
            waves[sid] = buffer.getvalue()
            for channel in (0, 1):
                rid = f'{sid}_ch{channel}'
                records.append({'id': rid, 'source_id': sid, 'channel_id': channel,
                    'l1': 'passive' if family == 'ship_noise' else 'active', 'l2': family, 'l3': leaf,
                    'fs': 16000, 'frames': 1600, 'duration_s': .1, 'signal_params': params,
                    'signal_frequency_hz': None if family == 'ship_noise' else 500.,
                    'G_h': -60., 'S_src': -10., 'audio_path': rid+'.wav'})
                received_metadata[rid] = {**source_metadata[sid], 'id': rid, 'source_id': sid,
                    'range_m': 999999, 'channel_model_version': 'fixture',
                    'bellhop_env': {'range_m': 1000., 'source_depth_m': 10., 'receiver_depth_m': 20., 'water_depth_m': 100.},
                    'bellhop_output': {'cir_delay_reference': 'first_arrival', 'absolute_first_arrival_delay_s': .6}}
    return records, source_metadata, received_metadata, waves


def add_member(stream, name, raw):
    member = tarfile.TarInfo(name); member.size = len(raw)
    stream.addfile(member, io.BytesIO(raw))


class ChannelStatisticsTests(unittest.TestCase):
    def test_selection_is_source_weighted_deterministic_and_requires_real_pairs(self):
        records, _, _, _ = fixtures()
        original = deepcopy(records)
        rows, proof = select_channel_pairs(records, CLASSES)
        self.assertEqual(rows, select_channel_pairs(list(reversed(records)), CLASSES)[0])
        self.assertEqual([r['source_id'] for r in rows], ['LFM_0', '2FSK_0', 'cruise_0'])
        self.assertEqual(proof['received_columns'], 2)
        self.assertTrue(proof['classes']['LFM']['dropped_zero_iqr'])
        single = [r for r in records if not (r['l3'] == 'cruise' and r['channel_id'] == 1)]
        rows, proof = select_channel_pairs(single, CLASSES)
        self.assertEqual(proof['received_columns'], 1)
        self.assertTrue(all(len(r['received']) == 1 for r in rows))
        self.assertEqual(records, original)

    def test_selection_rejects_duplicate_channels_and_inconsistent_source_features(self):
        records, _, _, _ = fixtures()
        for key, value in (('channel_id', 0), ('frames', 1599), ('signal_frequency_hz', 600)):
            bad = deepcopy(records); bad[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): select_channel_pairs(bad, CLASSES)

    def test_peak_scaling_preserves_structure_and_rejects_silence(self):
        audio = np.array([0., -2., 1., .5])
        scaled, proof = normalize_peak(audio)
        np.testing.assert_array_equal(audio, [0, -2, 1, .5])
        np.testing.assert_allclose(scaled, [0, -1, .5, .25])
        np.testing.assert_allclose(normalize_peak(audio*20)[0], scaled)
        self.assertEqual(proof['applied_scale'], .5)
        for invalid in ([], [0, 0], [0, float('nan')], [[1, 2]]):
            with self.assertRaises(ValueError): normalize_peak(invalid)

    def test_pair_identity_real_wav_headers_and_actual_channel_geometry(self):
        records, meta, received, waves = fixtures()
        row = select_channel_pairs(records, CLASSES)[0][0]; sid = row['source_id']
        header, geometry = validate_source_pair(row, meta[sid], waves[sid], received)
        self.assertEqual(header['duration_s'], .1)  # metadata incorrectly says 30 s
        self.assertEqual(geometry[0]['bellhop_env']['range_m'], 1000.)
        for case in ('id', 'parameters', 'alignment', 'sample_rate'):
            bad_source, bad_received, bad_row = deepcopy(meta[sid]), deepcopy(received), deepcopy(row)
            if case == 'id': bad_source['id'] = 'wrong'
            elif case == 'parameters': bad_source['signal_params']['center_freq_hz'] = 1000
            elif case == 'alignment': bad_received[row['received'][0]['id']]['bellhop_output']['cir_delay_reference'] = 'absolute'
            else: bad_row['received'][0]['fs'] = 48000
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_source_pair(bad_row, bad_source, waves[sid], bad_received)

    def test_tar_reader_requires_unique_complete_regular_members_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sources.tar'
            for case in ('valid', 'duplicate', 'missing', 'unsafe'):
                with tarfile.open(path, 'w') as stream:
                    add_member(stream, 'a/id.wav', b'wav')
                    if case != 'missing': add_member(stream, 'a/id.json', b'{}')
                    if case == 'duplicate': add_member(stream, 'b/id.wav', b'other')
                    if case == 'unsafe': add_member(stream, '../id.wav', b'unsafe')
                if case == 'valid':
                    self.assertEqual(read_source_members(path, ['id'])['id']['wav']['bytes'], b'wav')
                else:
                    with self.subTest(case=case), self.assertRaises(ValueError): read_source_members(path, ['id'])
                self.assertEqual(list(Path(directory).iterdir()), [path])


class ChannelPipelineTests(unittest.TestCase):
    def test_prepare_render_provenance_and_corruption_detection(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from visualization.plot_channel import build_channel_figure
        from visualization.project import paper_style
        records, sources, received, waves = fixtures()
        style = {'width_mm': 180, 'font_family': 'Times New Roman', 'font_size_pt': 9, 'legend_frame': False}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); archive = base/'sources.tar'
            with tarfile.open(archive, 'w') as stream:
                for sid in sources:
                    add_member(stream, sid+'.wav', waves[sid])
                    add_member(stream, sid+'.json', json.dumps(sources[sid]).encode())
            for record in records:
                target = base/(record['id']+'.wav'); target.write_bytes(waves[record['source_id']])
                record['audio_path'] = str(target)
            manifest = base/'eval.jsonl'
            manifest.write_text(''.join(json.dumps({'id': k, '_meta': v})+'\n' for k, v in received.items()))
            config_path = base/'channels.json'
            config_path.write_text(json.dumps({'cache_dir': 'cache', 'rows': [
                {'class': leaf, 'source_archive': str(archive)} for leaf in CLASSES]}))
            prepared = ({'eval_manifest': str(manifest), 'style': style}, [], records,
                        {'inputs_provenance_sha256': 'base-fixture'})
            with patch('visualization.channel_data.load_inputs', return_value=prepared), \
                    patch.dict(os.environ, {'UABENCH_CHANNEL_CONFIG': str(config_path)}):
                report = prepare_channels(); self.assertEqual(len(report['pairs']), 3)
                data, loaded_style, provenance = load_channels()
                self.assertIn('channels_provenance_sha256', provenance)
                original = deepcopy(data)
                with paper_style(loaded_style): fig, payload = build_channel_figure(data, loaded_style)
                try:
                    fig.canvas.draw(); json.dumps(payload, allow_nan=False)
                    self.assertEqual(data, original)
                    self.assertEqual(len(payload['panels']), 9)
                    np.testing.assert_allclose(fig.get_size_inches()*25.4, [180, 115])
                    for ax in fig.axes:
                        self.assertEqual(len(ax.get_shared_x_axes().get_siblings(ax)), 1)
                    for row in payload['rows']:
                        panels = [payload['panels'][sid] for sid in [row['source_id'], *row['received_ids']]]
                        self.assertEqual(len({json.dumps(p['time_window']) for p in panels}), 1)
                        self.assertEqual(len({p['power_reference'] for p in panels}), 1)
                        for panel in panels[1:]: np.testing.assert_allclose(panel['power_db'], panels[0]['power_db'])
                finally: plt.close(fig)
                source_wav = Path(data['rows'][0]['source']['audio_path'])
                source_wav.write_bytes(source_wav.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError, 'waveform changed'): load_channels()


if __name__ == '__main__':
    unittest.main()
