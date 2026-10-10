"""Audited source/received pairs for figure C; archive I/O only in preparation."""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

import numpy as np

from .acoustic_analysis import (PULSE_CLASSES, COMMUNICATION_CLASSES, SHIP_CLASSES,
                                load_audio, select_representatives)
from .project import ROOT, _write_gzip, _write_json, file_signature, load_inputs, sha256


def load_channel_config(path=None):
    path = Path(path or os.environ.get('UABENCH_CHANNEL_CONFIG', ROOT/'channels.local.json')).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Copy {ROOT / "channels.example.json"} to {path}, then prepare-channels')
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    rows = config['rows']
    if len(rows) != 3 or any(row['class'] not in family for row, family in
                            zip(rows, (PULSE_CLASSES, COMMUNICATION_CLASSES, SHIP_CLASSES))):
        raise ValueError('C requires one pulse, one communication and one ship class in that order')
    for row in rows:
        row['source_archive'] = str((path.parent/row['source_archive']).resolve())
    config['cache_dir'] = str((path.parent/config['cache_dir']).resolve())
    config['config_path'] = str(path)
    return config


def select_channel_pairs(records, classes):
    """Prefer two channels in Eval; fall back to one column for all rows if needed."""
    pools = {leaf: defaultdict(list) for leaf in classes}
    seen = set()
    for record in records:
        if record['l3'] not in pools:
            continue
        sid = record['source_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sid or ''):
            raise ValueError('Source ID must be a nonempty safe identifier')
        if record['id'] in seen:
            raise ValueError('Duplicate received record ID')
        seen.add(record['id'])
        pools[record['l3']][sid].append(record)
    for leaf, sources in pools.items():
        if not sources:
            raise ValueError(f'No Eval sources for {leaf}')
        for sid, channels in sources.items():
            ids = [r['channel_id'] for r in channels]
            if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in ids) or len(set(ids)) != len(ids):
                raise ValueError(f'{sid}: missing or duplicate channel ID')
            for key in ('l1', 'l2', 'l3', 'fs', 'frames', 'duration_s', 'signal_params', 'signal_frequency_hz'):
                if any(r[key] != channels[0][key] for r in channels):
                    raise ValueError(f'{sid}: inconsistent source property {key}')
    count = min(2, min(max(map(len, sources.values())) for sources in pools.values()))
    selected, proof = [], {}
    for leaf, sources in pools.items():
        candidates = []
        eligible = {sid: sorted(channels, key=lambda r: r['channel_id'])[:count]
                    for sid, channels in sources.items() if len(channels) >= count}
        for sid, channels in eligible.items():
            candidate = {**channels[0], 'id': sid}
            metric = 'S_src' if leaf in SHIP_CLASSES else 'G_h'
            values = [r[metric] for r in channels]
            if any(v is None or isinstance(v, bool) or not np.isfinite(v) for v in values):
                raise ValueError(f'{sid}: invalid selection metric {metric}')
            candidate[metric] = float(np.mean(values))
            candidates.append(candidate)
        chosen, details = select_representatives(candidates)
        sid = chosen[leaf]['id']
        selected.append({'class': leaf, 'source_id': sid, 'received': eligible[sid]})
        proof[leaf] = {**details['classes'][leaf], 'eval_sources': len(sources),
                       'eligible_sources': len(eligible), 'selected_channel_ids': [r['channel_id'] for r in eligible[sid]]}
    return selected, {'classes': proof, 'received_columns': count,
                     'method': 'one vote per eligible source; median/IQR distance using duration, mean selected-channel quality metric and active source frequency',
                     'channel_rule': 'up to two lowest channel IDs; two columns only when every row has an eligible pair',
                     'ties': 'lexicographic source ID; zero-IQR dimensions omitted',
                     'selection_uses_model_predictions': False}


def read_source_members(archive, source_ids):
    """Read only exact-basename WAV/JSON members; reject duplicate or linked matches."""
    found = {sid: {} for sid in source_ids}
    with tarfile.open(archive, 'r:*') as stream:
        for member in stream:
            path = PurePosixPath(member.name.replace('\\', '/'))
            if path.stem not in found or path.suffix.lower() not in ('.wav', '.json', '.jsonc'):
                continue
            kind = 'wav' if path.suffix.lower() == '.wav' else 'metadata'
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe or nonregular source member: {member.name}')
            if kind in found[path.stem]:
                raise ValueError(f'Ambiguous source member for {path.stem}: {kind}')
            if member.size > 100_000_000:
                raise ValueError(f'Unexpectedly large selected source member: {member.name}')
            with stream.extractfile(member) as handle:
                raw = handle.read()
            found[path.stem][kind] = {'bytes': raw, 'member': member.name,
                                      'size': member.size, 'sha256': hashlib.sha256(raw).hexdigest()}
    for sid, members in found.items():
        if set(members) != {'wav', 'metadata'}:
            raise ValueError(f'{sid}: source WAV and metadata both required in {archive}')
    return found


def validate_source_pair(row, source_meta, wav_bytes, received_meta):
    import soundfile as sf
    sid, leaf = row['source_id'], row['class']
    if source_meta.get('id') != sid:
        raise ValueError(f'{sid}: source metadata ID mismatch')
    if PurePosixPath(source_meta.get('wav_path', '').replace('\\', '/')).name != sid+'.wav':
        raise ValueError(f'{sid}: source metadata WAV identity mismatch')
    source_class = source_meta.get('sub_type') if leaf in SHIP_CLASSES else source_meta.get('signal_type')
    if source_class != leaf:
        raise ValueError(f'{sid}: source class mismatch')
    info = sf.info(io.BytesIO(wav_bytes))
    geometry = []
    for record in row['received']:
        meta = received_meta[record['id']]
        if meta.get('source_id') != sid or meta.get('id') != record['id']:
            raise ValueError(f'{sid}: received source ID mismatch')
        if source_meta.get('signal_params') != record['signal_params'] or meta.get('signal_params') != record['signal_params']:
            raise ValueError(f'{sid}: source/received generation parameters differ')
        if info.channels != 1 or info.samplerate != record['fs'] or info.frames != record['frames']:
            raise ValueError(f'{sid}: source and received WAV headers disagree; no implicit resampling/truncation')
        env, output = meta['bellhop_env'], meta['bellhop_output']
        for key in ('range_m', 'source_depth_m', 'receiver_depth_m', 'water_depth_m'):
            value = env.get(key)
            if value is None or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{record["id"]}: invalid bellhop_env.{key}')
        if output.get('cir_delay_reference') not in ('first_arrival', 'global_first_arrival'):
            raise ValueError(f'{record["id"]}: unknown received time alignment')
        delay = output.get('absolute_first_arrival_delay_s')
        if delay is None or not np.isfinite(delay) or delay < 0:
            raise ValueError(f'{record["id"]}: invalid arrival delay')
        geometry.append({'id': record['id'], 'channel_id': record['channel_id'],
                         'bellhop_env': env, 'bellhop_output': output,
                         'channel_model_version': meta.get('channel_model_version'),
                         'geometry_source': '_meta.bellhop_env, never top-level legacy range/depth fields'})
    return {'fs': info.samplerate, 'frames': info.frames, 'duration_s': info.frames/info.samplerate,
            'subtype': info.subtype, 'channels': info.channels}, geometry


def normalize_peak(samples):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError('Peak normalization needs a finite nonempty mono waveform')
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        raise ValueError('Cannot peak-normalize a silent waveform')
    return samples/peak, {'input_peak': peak, 'applied_scale': 1/peak, 'target_peak': 1.,
                          'domain': 'entire WAV before cropping; independently for each source/received WAV'}


def prepare_channels(config_path=None):
    config = load_channel_config(config_path)
    base, _, records, base_proof = load_inputs(include_full=False)
    rows, selection = select_channel_pairs(records, [r['class'] for r in config['rows']])
    wanted = {r['id'] for row in rows for r in row['received']}
    metadata = {}
    with Path(base['eval_manifest']).open(encoding='utf-8') as handle:
        for line in handle:
            item = json.loads(line)
            if item['id'] in wanted: metadata[item['id']] = item['_meta']
    archives = defaultdict(list)
    for setting, row in zip(config['rows'], rows):
        archives[setting['source_archive']].append(row['source_id'])
    members, signatures = {}, []
    for archive, source_ids in archives.items():
        print(f'Reading selected source members: {archive}', flush=True)
        before = file_signature(Path(archive))
        members.update(read_source_members(archive, source_ids))
        if file_signature(Path(archive)) != before: raise ValueError('Source archive changed during preparation')
        signatures.append(before)
    cache = Path(config['cache_dir']); cache.mkdir(parents=True, exist_ok=True)
    for setting, row in zip(config['rows'], rows):
        source = members[row['source_id']]
        # Current .jsonc source members are JSON-compatible. Reject unsupported syntax explicitly.
        source_meta = json.loads(source['metadata']['bytes'].decode('utf-8-sig'))
        header, geometry = validate_source_pair(row, source_meta, source['wav']['bytes'], metadata)
        target = cache/(row['source_id']+'.wav')
        target.write_bytes(source['wav']['bytes'])
        row['source'] = {**row['received'][0], **header, 'id': row['source_id'], 'channel_id': None,
                         'audio_path': str(target), 'audio_locator': {'kind': 'wav', 'path': str(target)}}
        row['source_origin'] = {'archive': setting['source_archive'],
                                **{kind: {k: v for k, v in part.items() if k != 'bytes'} for kind, part in source.items()},
                                'metadata': source_meta}
        row['geometry'] = geometry
        row['waveform_sha256'] = {}
        for record in [row['source'], *row['received']]:
            audio, proof = load_audio(record)
            normalize_peak(audio)
            row['waveform_sha256'][record['id']] = proof['audio_sha256']
    payload = {'rows': rows, 'selection': selection}
    output = cache/'pairs.json.gz'; _write_gzip(output, payload)
    receipt = {'schema': 1, 'config': config, 'base_inputs_sha256': base_proof['inputs_provenance_sha256'],
               'data': {'path': str(output), 'sha256': sha256(output)},
               'source_signatures': signatures,
               'code_sha256': {str(p): sha256(p) for p in (Path(__file__), ROOT/'acoustic_analysis.py')},
               'identity_checks': 'source metadata ID/class/parameters and WAV headers agree with audited received records; selected member SHA-256 retained',
               'archive_validation': 'size/mtime signatures of archives; SHA-256 of selected WAV and metadata members, not whole large archives'}
    _write_json(cache/'channels.provenance.json', receipt)
    return {'selection': selection, 'cache': str(cache), 'pairs': [
        {'source_id': r['source_id'], 'received_ids': [p['id'] for p in r['received']]} for r in rows]}


def load_channels():
    """Read prepared pairs and verify selected waveforms; no cache writes."""
    config = load_channel_config()
    base, _, _, base_proof = load_inputs(include_full=False)
    path = Path(config['cache_dir'])/'channels.provenance.json'
    if not path.is_file(): raise FileNotFoundError('C cache missing; run python -m visualization prepare-channels')
    proof = json.loads(path.read_text(encoding='utf-8'))
    if proof['schema'] != 1 or proof['config'] != config or proof['base_inputs_sha256'] != base_proof['inputs_provenance_sha256']:
        raise ValueError('Channel inputs/configuration changed; run prepare-channels')
    for signature in proof['source_signatures']:
        if file_signature(Path(signature['path'])) != signature:
            raise ValueError('Source archive changed; run prepare-channels')
    for source, expected in proof['code_sha256'].items():
        if sha256(Path(source)) != expected: raise ValueError('Channel analysis code changed; run prepare-channels')
    if sha256(Path(proof['data']['path'])) != proof['data']['sha256']:
        raise ValueError('Corrupt channel cache; run prepare-channels')
    with gzip.open(proof['data']['path'], 'rt', encoding='utf-8') as handle: data = json.load(handle)
    for row in data['rows']:
        for record in [row['source'], *row['received']]:
            if sha256(Path(record['audio_path'])) != row['waveform_sha256'][record['id']]:
                raise ValueError(f'Channel waveform changed: {record["id"]}; run prepare-channels')
    return data, base['style'], {**base_proof, 'channels': proof, 'channels_provenance_sha256': sha256(path)}
