"""
样本加载器 — 加载 Step 2 生成的 JSONL + 对应元数据 JSON，提取 L1/L2/L3 GT。
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from .parser import extract_l1_l2_l3_from_meta


@dataclass
class EvalSample:
    """单条评估样本。"""
    sample_id: str
    audio_path: str                       # 相对于 audio_root 的路径
    gt: Dict[str, str]                    # {"L1":"active","L2":"pulse","L3":"LFM"}
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLoader:
    """
    从 Step 2 生成的 JSONL + processed_audio 元数据加载评估样本。

    JSONL 格式 (Step 2 输出):
      {"id":"lfm_027998_ch0","audio":"PulseCom/wav_by_type/LFM/...","conversations":[...]}

    GT 来源: processed_audio/<dataset>/ 下的对应 JSON/JSONC 元数据文件。
    """

    def __init__(self, config: dict):
        pass

    def load(
        self,
        jsonl_path: str,
        audio_root: str,
        limit: int = None,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> List[EvalSample]:
        """加载评估样本。"""
        samples = []
        audio_base = Path(audio_root)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i % num_shards != shard_index:
                    continue
                if limit and len(samples) >= limit:
                    break
                try:
                    item = json.loads(line)
                    sample = self._build_sample(item, audio_base)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue
        return samples

    def _build_sample(self, item: dict, audio_base: Path) -> Optional[EvalSample]:
        sample_id = item.get("id", "")
        audio_rel = item.get("audio", "")

        if not sample_id or not audio_rel:
            return None

        # 优先使用内嵌 GT (JSONL 中直接带了 _gt 字段)
        embedded_gt = item.get("_gt")
        embedded_meta = item.get("_meta")
        if embedded_gt and embedded_gt.get("L1") != "unknown":
            gt = embedded_gt
            meta = {}
            if embedded_meta:
                for key in (
                    "tl_db", "snr_db", "quality_score", "quality_metric",
                    "quality_rank_within_l3", "quality_pool_size_within_l3",
                    "candidate_pool_size_within_l3", "selection_policy",
                ):
                    if key in embedded_meta:
                        meta[key] = embedded_meta[key]
                meta["ssp_complexity"] = embedded_meta.get("ssp_complexity")
        else:
            gt, meta = self._lookup_meta(audio_base, audio_rel, sample_id)

        if gt.get("L1") == "unknown":
            return None

        return EvalSample(
            sample_id=sample_id,
            audio_path=str(audio_base / audio_rel),
            gt=gt,
            metadata=meta,
        )

    def _lookup_meta(
        self, audio_base: Path, audio_rel: str, sample_id: str
    ) -> tuple:
        """从 processed_audio 元数据 JSON 中提取 GT + SNR/SSP 元数据。"""
        json_candidates = []
        audio_path = Path(audio_rel)
        parts = audio_path.parts

        if "PulseCom" in audio_rel and len(parts) >= 3:
            sig_type = parts[-2]
            json_rel = Path("PulseCom") / "jsonc" / sig_type / f"{sample_id}.jsonc"
            json_candidates.append(audio_base / json_rel)
        elif "05_ship_radiated_noise" in audio_rel and len(parts) >= 3:
            class_name = parts[-3]
            json_rel = Path("05_ship_radiated_noise") / class_name / "json" / f"{sample_id}.json"
            json_candidates.append(audio_base / json_rel)

        for ext in [".jsonc", ".json"]:
            pattern = f"{sample_id}{ext}"
            for m in audio_base.rglob(pattern):
                if m not in json_candidates:
                    json_candidates.append(m)

        for jp in json_candidates:
            if jp.exists():
                try:
                    raw = json.loads(jp.read_text(encoding="utf-8"))
                    gt = extract_l1_l2_l3_from_meta(raw)
                    if gt.get("L1") == "unknown":
                        continue

                    # 提取质量指标: PulseCom → tl_db, Ship → snr_db
                    bo = raw.get("bellhop_output", {})
                    metadata = {}
                    tl = bo.get("tl_db")
                    snr = bo.get("snr_db")
                    if tl is not None:
                        metadata["tl_db"] = float(tl)
                    if snr is not None:
                        metadata["snr_db"] = float(snr)

                    # 提取 SSP 复杂度 (声速梯度方差, 用于分层)
                    be = raw.get("bellhop_env", {})
                    ssp_data = be.get("ssp", {})
                    depths = ssp_data.get("depths_m")
                    speeds = ssp_data.get("sound_speeds_mps")
                    if depths and speeds and len(depths) > 1:
                        gradients = [
                            (speeds[i+1] - speeds[i]) / (depths[i+1] - depths[i])
                            for i in range(len(depths) - 1)
                        ]
                        mean_grad = sum(gradients) / len(gradients)
                        complexity = sum((g - mean_grad) ** 2 for g in gradients) / len(gradients)
                        metadata["ssp_complexity"] = round(complexity, 8)

                    return gt, metadata
                except (json.JSONDecodeError, Exception):
                    continue

        print(f"  [WARN] GT lookup failed for {sample_id}, audio={audio_rel}")
        return {"L1": "unknown", "L2": "unknown", "L3": "unknown"}, {}
