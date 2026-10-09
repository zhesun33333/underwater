# 首批图注与正文观察草案

下面文字对应当前真实数据输出。图号由论文排版决定，文件名不绑定最终编号。

## B — selection_distributions.pdf

**Caption.** Record-weighted empirical cumulative distributions for the full test partition (dashed lines) and UA-Bench-Eval (solid lines): (a) duration measured from WAV headers, (b) active-signal center or carrier frequency, (c) pre-normalization channel energy gain, $G_h$, and (d) source tonal-to-continuum power ratio, $S_{\mathrm{src}}$, for ship noise. Colors indicate signal families. Panel sample counts are reported in full-test/Eval order; no applicable values are missing. Separate channel realizations are counted as separate records.

可据实写入正文的观察：评测子集的主动信号频率和筛选质量统计量明显向较高值移动。Pulse 的中心频率中位数从 3327.7 Hz 变为 4919.0 Hz，Communication 的载频中位数从 2623.3 Hz 变为 4660.4 Hz；两家族的 $G_h$ 中位数分别从 −117.71、−104.24 dB 变为 −78.04、−70.72 dB。舰船 $S_{\mathrm{src}}$ 中位数从 −13.34 dB 变为 −10.59 dB。这描述的是子集筛选后的输入分布，不是模型性能对信道条件的因果效应。

## A1 — acoustic_active.pdf

**Caption.** Received-waveform structure for one deterministically selected example from each active leaf class: (a–c) complete pulse inputs, (d) OFDM, and (e–h) centered windows of up to 24 nominal symbols for the four single-carrier modulations. Examples minimize the Euclidean distance to the within-class median vector after IQR scaling of actual duration, $G_h$, and signal center/carrier frequency; IDs break ties. Pulse STFTs use a 16-ms Hann window and 2-ms hop. Single-carrier windows span two symbols with a one-eighth-symbol hop; OFDM uses a 32-ms window and 4-ms hop. Each family's color scale has its own common power reference, defined by the maximum over its selected analysis windows. Panels (e–h) show milliseconds from the crop start; the other panels show received-WAV seconds. Power views characterize time-frequency structure and do not establish PSK phase-state counts.

阅读边界：零填充只加密频率网格，不提高真实频率分辨率；实际窗长、FFT 点数和等效噪声带宽在伴随 JSON 中。两条色条的 0 dB 参考值不同，不能据跨家族亮度比较物理声级。OFDM 示例只有 4 个符号，所以按同一固定规则显示完整 2.56 s 输入。

## A2 — acoustic_ships.pdf

**Caption.** Ship-noise spectra and population summaries. (a–e) Welch power spectral densities for one deterministically selected received example from each of the five ship-noise classes. Selection uses within-class IQR-scaled duration and $S_{\mathrm{src}}$. These five panels share frequency and level axes. (f) Band-limited spectral centroids for all 1,000 ship-noise records in UA-Bench-Eval (200 per class): points show medians and bars show the 25th–75th percentiles, not confidence intervals. All estimates use the entire received waveform with a 1-s Hann window, 50% overlap, constant detrending and mean averaging. Centroids are computed as $\sum_k f_k P_k/\sum_k P_k$ over Welch bins in 0–1000 Hz, using linear power density $P_k$. PSD levels in (a–e) are expressed as $10\log_{10}[\mathrm{PSD}/(1\ \mathrm{digital\ amplitude}^{2}/\mathrm{Hz})]$, with no additional gain normalization. Channel realizations count as separate records. The displayed frequency interval does not imply an absence of higher-frequency energy.

前五幅是确定规则选出的单条示例，不代表该类所有样本共有的谱形。第六幅统计各类全部 Eval 记录在 0–1000 Hz 内的功率加权平均频率，不能解释为全频带谱质心、置信区间或类别可分性检验。音频 ID、source ID、SHA-256、每条谱质心及分位数均在伴随数据中。
