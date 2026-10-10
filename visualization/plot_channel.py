"""Figure C: independent source/received axes with explicit structural scaling."""
from __future__ import annotations

import numpy as np

from .acoustic_analysis import (SHIP_CLASSES, communication_window, compute_spectrogram,
                                compute_welch, frequency_band, load_audio, power_db,
                                stft_parameters, welch_parameters)
from .channel_data import normalize_peak
from .plot_acoustic import FAMILY_COLORS, SHIP_NAMES, _axis, _image_extent


def build_channel_figure(data, style):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator
    from cmcrameri import cm

    rows = data['rows']
    if len(rows) != 3 or len({len(r['received']) for r in rows}) != 1:
        raise ValueError('C requires three rows with a consistent received-column count')
    columns = len(rows[0]['received'])+1
    if columns not in (2, 3): raise ValueError('C supports source plus one or two received columns')
    width = style['width_mm']
    height = style.get('channel_height_mm', width*23/36)
    if not np.isfinite(height) or height <= 0: raise ValueError('Invalid channel_height_mm')
    # Reserve physical space for 9-pt titles/labels rather than scaling the whitespace.
    bottom_mm, top_mm, row_gap_mm = 20., 11., 17.
    panel_height_mm = (height-bottom_mm-top_mm-2*row_gap_mm)/3
    if panel_height_mm <= 0: raise ValueError('channel_height_mm leaves no space for the panels')
    fig = plt.figure(figsize=(width/25.4, height/25.4), dpi=300)
    grid = fig.add_gridspec(3, columns, left=.10, right=.89, bottom=bottom_mm/height,
                           top=1-top_mm/height, wspace=.59, hspace=row_gap_mm/panel_height_mm)
    panels, references = {}, {}
    floor = -50.
    for row_index, row in enumerate(rows):
        leaf = row['class']; ship = leaf in SHIP_CLASSES
        records = [row['source'], *row['received']]
        if any(r['source_id'] != row['source_id'] for r in records):
            raise ValueError('C cannot mix different source identities within a row')
        loaded = [load_audio(r) for r in records]
        if len({(p['fs'], p['frames']) for _, p in loaded}) != 1:
            raise ValueError('C cannot silently resample or truncate source/received audio')
        normalized = [normalize_peak(a) for a, _ in loaded]
        first = records[0]
        if first['l2'] == 'communication':
            crop = communication_window(first)
        else:
            crop = {'start_sample': 0, 'stop_sample_exclusive': first['frames'], 'start_s': 0.,
                    'stop_s': first['duration_s'], 'used_entire_waveform': True}
        parameters = welch_parameters(first['fs'], first['frames']) if ship else stft_parameters(first, crop)
        spectra = []
        for audio, _ in normalized:
            if ship:
                f, values = compute_welch(audio, parameters)
                keep = f <= min(1000., first['fs']/2)
                spectra.append((f[keep], None, values[keep]))
            else:
                f, t, values = compute_spectrogram(audio[crop['start_sample']:crop['stop_sample_exclusive']],
                                                  parameters, crop['start_s'])
                spectra.append((f, t, values))
        reference = 1. if ship else max(float(np.max(s[2])) for s in spectra)
        references[leaf] = reference
        band = None if ship else frequency_band(first, parameters['fft_bin_spacing_hz'])
        db_values = [power_db(s[2], reference, -120. if ship else floor) for s in spectra]
        if ship:
            y_limits = [10*np.floor(min(float(np.min(db)) for db in db_values)/10),
                        10*np.ceil(max(float(np.max(db)) for db in db_values)/10)]
            if y_limits[0] == y_limits[1]: y_limits[0] -= 10
        row_axes = []
        for col, (record, (frequencies, times, values), db, (_, audio_proof), (_, norm_proof)) in enumerate(
                zip(records, spectra, db_values, loaded, normalized)):
            ax = fig.add_subplot(grid[row_index, col]); row_axes.append(ax)
            local = first['l2'] == 'communication'
            if ship:
                ax.plot(frequencies, db, color=FAMILY_COLORS['ship_noise'], linewidth=.75)
                ax.set_xlim(0, min(1000., first['fs']/2)); ax.set_ylim(y_limits)
                ax.set_xticks([0, 500, 1000])
                ax.set_xlabel('Frequency (Hz)', labelpad=2)
                ax.set_ylabel('PSD (dB/Hz)', labelpad=2)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                low, high = band['display_limits_hz']
                step = parameters['fft_bin_spacing_hz']
                keep = (frequencies >= low-step) & (frequencies <= high+step)
                frequencies, values, db = frequencies[keep], values[keep], db[keep]
                extent = _image_extent(frequencies, times, step, parameters['hop_duration_s'])
                origin, scale = (crop['start_s'], 1000.) if local else (0., 1.)
                extent = [(extent[0]-origin)*scale, (extent[1]-origin)*scale, *extent[2:]]
                ax.imshow(db, origin='lower', aspect='auto', interpolation='nearest', extent=extent,
                          cmap=cm.batlow, vmin=floor, vmax=0)
                limits = [(crop['start_s']-origin)*scale, (crop['stop_s']-origin)*scale]
                ax.set_xlim(limits); ax.set_ylim(band['display_limits_hz'])
                ax.set_xticks(np.linspace(*limits, 3))
                ax.xaxis.set_major_formatter(lambda value, _: f'{value:.3g}')
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
                ax.set_xlabel('Window time (ms)' if local else 'Aligned time (s)', labelpad=2)
                ax.set_ylabel('Frequency (Hz)', labelpad=2)
            label = SHIP_NAMES[leaf].replace(' ship', '') if ship else leaf
            condition = 'Source' if col == 0 else f'ch{record["channel_id"]}'
            detail = 'Unpropagated' if col == 0 else f'r = {row["geometry"][col-1]["bellhop_env"]["range_m"]/1000:.2f} km'
            ax.set_title(f'({chr(97+row_index*columns+col)}) {label} · {condition}\n{detail}', pad=5,
                         color=FAMILY_COLORS[first['l2']])
            ax.get_xticklabels()[0].set_ha('left'); ax.get_xticklabels()[-1].set_ha('right')
            _axis(ax)
            if col == 0:
                audio_proof['channel_frequency_hz'] = None  # Unpropagated source has no Bellhop frequency.
            panels[record['id']] = {**audio_proof, 'additional_gain_normalization': True,
                'normalization': norm_proof, 'row': row_index, 'column': col,
                'time_window': crop, 'analysis': dict(parameters), 'frequency_band': band,
                'frequency_hz': frequencies.tolist(), 'time_s': None if times is None else times.tolist(),
                'power_linear': values.tolist(), 'power_db': db.tolist(), 'power_reference': reference,
                'display_time_origin_s': crop['start_s'] if local else 0.,
                'display_time_unit': 'ms' if local else 's',
                'source_origin': row['source_origin'] if col == 0 else None,
                'channel_metadata': None if col == 0 else row['geometry'][col-1]}
        if not ship:
            position = row_axes[-1].get_position()
            cax = fig.add_axes([.915, position.y0, .014, position.height])
            bar = fig.colorbar(ScalarMappable(norm=Normalize(floor, 0), cmap=cm.batlow), cax=cax)
            bar.solids.set_rasterized(False)
            bar.set_ticks([-50, -25, 0]); bar.set_label('Power (dB)', labelpad=3)
            bar.outline.set_linewidth(.75)
            cax.tick_params(direction='in', width=.75, length=3); cax.minorticks_off()
    fig.text(.5, 8.1/height, 'Each WAV peak-normalized to 1; STFT 0 dB = maximum within each row.', ha='center')
    fig.text(.5, 2.97/height, 'Received time is arrival-aligned; these panels compare structure, not propagation loss.', ha='center')
    return fig, {'figure': 'channel_examples', 'selection': data['selection'],
        'layout': f'3 rows x {columns} independent axes: source followed by received channels',
        'rows': [{'class': r['class'], 'source_id': r['source_id'],
                  'received_ids': [x['id'] for x in r['received']]} for r in rows],
        'panels': panels, 'row_power_references': references, 'stft_display_range_db': [floor, 0.],
        'normalization': 'each complete WAV independently divided by its absolute peak before identical within-row cropping',
        'time_alignment': 'received WAVs already aligned to first arrival by propagation pipeline; no additional shifting; source time zero retained',
        'communication_crop': 'same centered window of up to 24 nominal symbols for source and received WAVs; not recovered symbol alignment',
        'stft_reference': 'one-sided squared spectrum-scaled STFT, referenced to row maximum over source and selected received analysis windows and all frequency bins',
        'ship_psd_reference': 'Welch density of dimensionless peak-normalized samples; dB relative to 1/Hz; common row limits, full WAV, 0-1000 Hz shown',
        'interpretation': 'three illustrative matched sources; no physical SPL/transmission loss or class-wide claims',
        'raster_dpi_minimum': 300}
