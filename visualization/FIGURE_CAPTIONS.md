# 论文图注与正文观察草案

下面文字对应当前真实数据输出；D–G 使用 2026-10-10 的九模型归档及配对静音结果。图号由论文排版决定，文件名不绑定最终编号。

## B — selection_distributions.pdf

**Caption.** Record-weighted empirical cumulative distributions for the full test partition (dashed lines) and UA-Bench-Eval (solid lines): (a) duration measured from WAV headers, (b) active-signal center or carrier frequency, (c) pre-normalization channel energy gain, $G_h$, and (d) source tonal-to-continuum power ratio, $S_{\mathrm{src}}$, for ship noise. Colors indicate signal families. Panel sample counts are reported in full-test/Eval order; no applicable values are missing. Separate channel realizations are counted as separate records.

可据实写入正文的观察：评测子集的主动信号频率和筛选质量统计量明显向较高值移动。Pulse 的中心频率中位数从 3327.7 Hz 变为 4919.0 Hz，Communication 的载频中位数从 2623.3 Hz 变为 4660.4 Hz；两家族的 $G_h$ 中位数分别从 −117.71、−104.24 dB 变为 −78.04、−70.72 dB。舰船 $S_{\mathrm{src}}$ 中位数从 −13.34 dB 变为 −10.59 dB。这描述的是子集筛选后的输入分布，不是模型性能对信道条件的因果效应。

## A1 — acoustic_active.pdf

**Caption.** Received-waveform structure for one deterministically selected example from each active leaf class: (a–c) complete pulse inputs, (d) OFDM, and (e–h) centered windows of up to 24 nominal symbols for the four single-carrier modulations. Examples minimize the Euclidean distance to the within-class median vector after IQR scaling of actual duration, $G_h$, and signal center/carrier frequency; IDs break ties. Pulse STFTs use a 16-ms Hann window and 2-ms hop. Single-carrier windows span two symbols with a one-eighth-symbol hop; OFDM uses a 32-ms window and 4-ms hop. Each family's color scale has its own common power reference, defined by the maximum over its selected analysis windows. Panels (e–h) show milliseconds from the crop start; the other panels show received-WAV seconds. Power views characterize time-frequency structure and do not establish PSK phase-state counts.

阅读边界：零填充只加密频率网格，不提高真实频率分辨率；实际窗长、FFT 点数和等效噪声带宽在伴随 JSON 中。两条色条的 0 dB 参考值不同，不能据跨家族亮度比较物理声级。OFDM 示例只有 4 个符号，所以按同一固定规则显示完整 2.56 s 输入。

## A2 — acoustic_ships.pdf

**Caption.** Ship-noise spectra and population summaries. (a–e) Welch power spectral densities for one deterministically selected received example from each of the five ship-noise classes. Selection uses within-class IQR-scaled duration and $S_{\mathrm{src}}$. These five panels share frequency and level axes. (f) Band-limited spectral centroids for all 1,000 ship-noise records in UA-Bench-Eval (200 per class): points show medians and bars show the 25th–75th percentiles, not confidence intervals. All estimates use the entire received waveform with a 1-s Hann window, 50% overlap, constant detrending and mean averaging. Centroids are computed as $\sum_k f_k P_k/\sum_k P_k$ over Welch bins in 0–1000 Hz, using linear power density $P_k$. PSD levels in (a–e) are expressed as $10\log_{10}[\mathrm{PSD}/(1\ \mathrm{digital\ amplitude}^{2}/\mathrm{Hz})]$, with no additional gain normalization. Channel realizations count as separate records. The displayed frequency interval does not imply an absence of higher-frequency energy.

前五幅是确定规则选出的单条示例，不代表该类所有样本共有的谱形。第六幅统计各类全部 Eval 记录在 0–1000 Hz 内的功率加权平均频率，不能解释为全频带谱质心、置信区间或类别可分性检验。音频 ID、source ID、SHA-256、每条谱质心及分位数均在伴随数据中。

## C — channel_examples.pdf

**Caption.** Source and propagated waveforms for three matched sources: LFM detection pulse, 2FSK communication signal and cruise-ship noise. Columns show the unpropagated source and two channel realizations present in UA-Bench-Eval; channel IDs and propagation distances are indicated. Each complete WAV is independently normalized to unit peak before analysis. Active rows show STFT power relative to the maximum within that row, using identical windows and frequency limits across its three columns. The pulse is shown in full; the communication panels use the same centered interval of up to 24 nominal symbols, with milliseconds measured from its start. Received time axes follow the stored first-arrival alignment; no additional time shift is applied. The ship row shows full-waveform Welch PSDs over 0–1000 Hz using a 1-s Hann window and 50% overlap, expressed in dB relative to 1/Hz for dimensionless normalized samples. Examples are selected by a source-weighted median/IQR metadata rule among sources with paired Eval channels. The panels illustrate structural changes for these sources, not physical propagation loss or class-wide effects.

不同主动行的 0 dB 参考独立。全部波形采用同一种“完整波形峰值归一化”规则，不能据源与输出的亮度或谱高差估算传输损失。所选 source ID、音频哈希、窗口、缩放系数、实际传播环境与到达时延均在伴随记录中。

## D — class_diagnostics.pdf

**Caption.** Class diagnostics for the Qwen2.5-Omni original-audio run in the 2026-10-10 results archive (2,600 records). This illustrative model has nonzero correct leaf predictions in both active-signal and ship-noise groups; it is not an average over models. (a) Row-normalized leaf-class confusion, retaining unresolved and cascaded predictions in each true-class denominator. Darker and larger rounded squares indicate higher shares; neutral dots indicate zero observations. Color is linear over 0–100%. Square side length follows 0.45 + 0.50 sqrt(p) in cell-width units for a positive row fraction p, so area is not proportional to probability. (b) Precision, recall and F1 computed from the same stored confusion counts. Cell labels show percentages of at least 10%. Predictions, saved metrics, the archived manifest and protocol shards were audited without rerunning inference.

该图对应最新逐样本结果。舰船预测仍明显偏向图中的 Naval（内部类别为 Warship）：1,000 条舰船样本中有 884 条被预测为该类，因此不能把这张单模型图理解为类别均衡的表现。归档协议与本地代码的哈希比较范围及音频路径差异见 [RESULTS_GUIDE.md](RESULTS_GUIDE.md)；“归档核验”不等同于按当前本地推理代码重新运行。

## E — error_decomposition.pdf

**Caption.** Outcome composition for nine archived original-audio model runs (2,600 records per model). Five mutually exclusive outcomes are shown: invalid Turn 1 format, valid but incorrect Turn 1 routing, invalid Turn 2 format, valid but incorrect leaf prediction after Turn 2, and correct leaf prediction. Explicit saved parse statuses distinguish format failures; all records contribute to each model's denominator. Numbers inside sufficiently large segments are percentages.

五类结局按各模型保存的逐样本状态计算，合计均为 2,600；本图不将旧包缺失的格式状态推断为“有效”。

## F — metadata_performance.pdf

**Caption.** Class-balanced leaf recall across fixed, within-class metadata tertiles for nine archived original-audio runs: (a) active signals stratified by pre-normalization channel energy gain, $G_h$; (b) ship noise stratified by source tonal-to-continuum power ratio, $S_{\mathrm{src}}$. Strata are computed once from the audited Eval manifest and shared by all models. Recall is averaged equally across eight active or five ship classes, with unresolved and cascaded predictions counted as errors. Colors, line styles and markers identify models. Shading spans the observed minimum and maximum across the nine displayed models at each stratum; Δ gives their difference in percentage points. Extrema may belong to different models across strata. The band is neither a confidence interval nor a theoretical bound; connecting lines join ordered stratum summaries rather than estimate continuous-condition performance. Equal metadata values remain together; a group-stratum with any missing class is not averaged over a reduced class set, and bounds are omitted wherever any displayed model is missing. The figure shows associations, not effects of changing propagation conditions.

## G — silence_control.pdf

**Caption.** Paired original-audio and matched-silence evaluations for nine archived model runs, with 2,600 matched sample IDs per model. Connected points show (a) leaf-level L3 accuracy and (b) broad L1 accuracy. Signed labels report original minus silent accuracy in percentage points; negative and zero differences are retained. Saved pair records, original and silent predictions, each condition's manifest and protocol, input hashes, and reported counts were cross-checked. The two manifests agree on every non-audio field. The archive contains no WAV files, so the silence waveform bytes could not be independently rechecked. The points describe these archived runs and do not represent confidence intervals or newly executed inference.

静音对照的差值并非对所有模型同向：L3 差值介于 −1.42 与 +4.73 个百分点，L1 差值介于 −0.12 与 +29.54 个百分点。这里仅描述九次保存运行，不作显著性或跨运行稳定性的推断。
