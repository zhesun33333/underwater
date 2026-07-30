"""模型推理统一接口。

支持后端:
  - mock:             模拟输出, 测试框架
  - qwen2_audio:      Qwen2-Audio-7B-Instruct
  - aero1_audio:      Aero-1-Audio (1.5B, LMMs-Lab)
  - voxtral_mini:     Voxtral-Mini-3B-2507 (Mistral)
  - af_next:          Audio-Flamingo-Next (7B, NVIDIA)
  - kimi_audio:       Kimi-Audio-7B-Instruct (Moonshot, ~10B)
"""
import random


class ModelInference:
    """统一推理接口，根据 config['model']['backend'] 分发到具体后端。"""

    BACKENDS = {}  # populated at module bottom

    def __init__(self, config: dict):
        model_cfg = config["model"]
        self.backend = model_cfg.get("backend", "mock")
        cls = self.BACKENDS.get(self.backend)
        if cls is None:
            raise ValueError(f"Unknown backend: {self.backend}. Available: {list(self.BACKENDS)}")
        self._model = cls(model_cfg)

    def generate(self, audio_path: str, prompt: str,
                 max_new_tokens: int = 512, temperature: float = 0.1,
                 no_audio: bool = False) -> str:
        return self._model.generate(audio_path, prompt, max_new_tokens,
                                     temperature, no_audio=no_audio)

    def chat(self, audio_path: str, conversations: list,
             max_new_tokens: int = 512, temperature: float = 0.1,
             no_audio: bool = False) -> str:
        return self._model.chat(audio_path, conversations, max_new_tokens,
                                 temperature, no_audio=no_audio)


# ============================================================
# 共享推理逻辑 (generate / chat 的公共部分)
# ============================================================
class _BaseAudioBackend:
    """音频 LLM 后端基类。子类只需实现 _load_model 和 _load_processor。"""

    _default_sampling_rate: int = 16000

    def __init__(self, model_cfg: dict):
        import torch

        self.model_id = model_cfg.get("model_id")
        if not self.model_id:
            raise ValueError(f"{self.__class__.__name__}: model_id is required")

        self.device = model_cfg.get("device", "cuda:0")
        self.torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
        self.max_new_tokens = model_cfg.get("max_new_tokens", 256)
        self.default_temperature = model_cfg.get("temperature", 0.0)

        print(f"  Loading {self.__class__.__name__}: {self.model_id}")
        print(f"  device={self.device}, dtype={self.torch_dtype}")

        self.model = self._load_model()
        self.processor = self._load_processor()
        self.sampling_rate = getattr(
            getattr(self.processor, "feature_extractor", None), "sampling_rate",
            self._default_sampling_rate,
        )
        print(f"  Model loaded, sample_rate={self.sampling_rate}Hz")

    def _load_model(self):
        raise NotImplementedError

    def _load_processor(self):
        raise NotImplementedError

    # ---- generate / chat ----
    def generate(self, audio_path: str, prompt: str,
                 max_new_tokens: int = None, temperature: float = None,
                 no_audio: bool = False) -> str:
        max_t = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.default_temperature
        return self._infer(audio_path, prompt, None, max_t, temp, no_audio)

    def chat(self, audio_path: str, conversations: list,
             max_new_tokens: int = None, temperature: float = None,
             no_audio: bool = False) -> str:
        max_t = max_new_tokens or self.max_new_tokens
        temp = temperature if temperature is not None else self.default_temperature
        return self._infer(audio_path, None, conversations, max_t, temp, no_audio)

    def _infer(self, audio_path, prompt, conversations,
               max_new_tokens, temperature, no_audio):
        """统一推理入口: 构建 conversation → processor → model.generate → decode。"""
        import torch
        import librosa

        # 1. 构建 conversation
        if conversations is not None:
            conv = []
            for i, turn in enumerate(conversations):
                role = "user" if turn["from"] == "human" else "assistant"
                if i == 0 and role == "user":
                    content = [{"type": "text", "text": turn["value"]}]
                    if not no_audio:
                        content.insert(0, {"type": "audio", "audio_url": audio_path})
                    conv.append({"role": "user", "content": content})
                elif role == "user":
                    conv.append({"role": "user", "content": [{"type": "text", "text": turn["value"]}]})
                else:
                    conv.append({"role": "assistant", "content": turn["value"]})
        else:
            content = [{"type": "text", "text": prompt}]
            if not no_audio:
                content.insert(0, {"type": "audio", "audio_url": audio_path})
            conv = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False)

        # 2. Processor
        if no_audio:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        else:
            audio_array, _ = librosa.load(audio_path, sr=self.sampling_rate)
            inputs = self._call_processor(text, audio_array)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

        # 3. Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # 4. Decode
        return self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    def _call_processor(self, text: str, audio_array):
        """调用 processor 处理 text + audio。子类可重写以适配不同 API。
        默认: audios=[audio_array] (Qwen2-Audio 风格)"""
        return self.processor(
            text=text, audios=[audio_array],
            return_tensors="pt", sampling_rate=self.sampling_rate,
        )


# ============================================================
# Qwen2-Audio-7B
# ============================================================
class Qwen2AudioBackend(_BaseAudioBackend):
    """Qwen2-Audio-7B-Instruct (Alibaba)"""

    def _load_model(self):
        from transformers import Qwen2AudioForConditionalGeneration
        return Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.startswith("cuda") else self.device,
            low_cpu_mem_usage=True,
        )

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id)


# ============================================================
# Aero-1-Audio (1.5B, LMMs-Lab)
# https://huggingface.co/lmms-lab/Aero-1-Audio
# ============================================================
class Aero1AudioBackend(_BaseAudioBackend):
    """Aero-1-Audio by LMMs-Lab. Qwen2.5-1.5B backbone, 50k hrs curated audio."""

    def _load_model(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.startswith("cuda") else self.device,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)


# ============================================================
# Voxtral-Mini-3B (Mistral AI)
# https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
# ============================================================
class VoxtralMiniBackend(_BaseAudioBackend):
    """Voxtral-Mini-3B by Mistral. apply_chat_template 直接返回 tensor。"""

    def _load_model(self):
        from transformers import VoxtralForConditionalGeneration
        return VoxtralForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.startswith("cuda") else self.device,
            low_cpu_mem_usage=True,
        )

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id)

    def _infer(self, audio_path, prompt, conversations,
               max_new_tokens, temperature, no_audio):
        """Voxtral: apply_chat_template 一次性处理 audio+text, 返回 inputs tensor."""
        import torch

        # 构建 conversation (Voxtral 用 "path" 而非 "audio_url")
        if conversations is not None:
            conv = []
            for i, turn in enumerate(conversations):
                role = "user" if turn["from"] == "human" else "assistant"
                if i == 0 and role == "user":
                    content = [{"type": "text", "text": turn["value"]}]
                    if not no_audio:
                        content.insert(0, {"type": "audio", "path": audio_path})
                    conv.append({"role": "user", "content": content})
                elif role == "user":
                    conv.append({"role": "user", "content": [{"type": "text", "text": turn["value"]}]})
                else:
                    conv.append({"role": "assistant", "content": turn["value"]})
        else:
            content = [{"type": "text", "text": prompt}]
            if not no_audio:
                content.insert(0, {"type": "audio", "path": audio_path})
            conv = [{"role": "user", "content": content}]

        # apply_chat_template 直接返回 inputs tensor (不需要 tokenize=False)
        inputs = self.processor.apply_chat_template(conv)
        inputs = inputs.to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        return self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0].strip()


# ============================================================
# Audio-Flamingo-Next (7B, NVIDIA)
# https://huggingface.co/nvidia/Audio-Flamingo-Next-Instruct
# ============================================================
class AudioFlamingoNextBackend(_BaseAudioBackend):
    """Audio-Flamingo-Next by NVIDIA. Best open non-speech audio model."""

    def _load_model(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype,
            device_map=self.device if self.device.startswith("cuda") else self.device,
            low_cpu_mem_usage=True, trust_remote_code=True,
        )

    def _load_processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def _call_processor(self, text: str, audio_array):
        """AF-Next: try audios first, fall back to audio."""
        try:
            return self.processor(
                text=text, audios=[audio_array],
                return_tensors="pt", sampling_rate=self.sampling_rate,
            )
        except TypeError:
            return self.processor(
                text=text, audio=audio_array,
                return_tensors="pt", sampling_rate=self.sampling_rate,
            )


# ============================================================
# Kimi-Audio-7B-Instruct (Moonshot AI, ~10B)
# https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
# ============================================================
class KimiAudioBackend(_BaseAudioBackend):
    """Kimi-Audio-7B-Instruct by Moonshot AI. 13M hrs pretraining.

    使用官方 kimia_infer API, 不走 transformers processor 模式。
    需要: pip install kimia-infer (或从 GitHub 安装)
    """

    def _load_model(self):
        try:
            from kimia_infer.api.kimia import KimiAudio
        except ImportError:
            raise ImportError(
                "Kimi-Audio 需要安装 kimia-infer 包.\n"
                "  pip install git+https://github.com/MoonshotAI/Kimi-Audio.git\n"
                "或从 ModelScope 下载后本地安装: pip install -e /path/to/Kimi-Audio"
            )
        self._kimia_params = {
            "audio_temperature": 0.0,
            "text_temperature": self.default_temperature,
            "output_type": "text",
        }
        return KimiAudio(model_path=self.model_id, load_detokenizer=False)

    def _load_processor(self):
        return None  # Kimi-Audio 不需要 processor, 模型自带一切

    def _infer(self, audio_path, prompt, conversations,
               max_new_tokens, temperature, no_audio):
        """Kimi-Audio 不走标准 generate/processor 流程, 重写整个推理逻辑。"""
        # 构建 Kimi-Audio 格式的 messages
        if conversations is not None:
            messages = []
            for turn in conversations:
                role = turn["from"] == "human" and "user" or "assistant"
                if role == "user":
                    messages.append({
                        "role": "user", "message_type": "text",
                        "content": turn["value"],
                    })
                    if turn == conversations[0] and not no_audio:
                        messages.append({
                            "role": "user", "message_type": "audio",
                            "content": audio_path,
                        })
                else:
                    messages.append({
                        "role": "assistant", "message_type": "text",
                        "content": turn["value"],
                    })
        else:
            messages = [{"role": "user", "message_type": "text", "content": prompt}]
            if not no_audio:
                messages.append({"role": "user", "message_type": "audio", "content": audio_path})

        params = dict(self._kimia_params)
        if temperature is not None:
            params["text_temperature"] = temperature
        params["output_type"] = "text"

        _, text_output = self.model.generate(messages, **params)
        return (text_output or "").strip()


# ============================================================
# MockModel (测试用)
# ============================================================
class MockModel:
    """模拟模型，用于测试评估框架。"""

    L1_MAP = {"active": "主动信号", "passive": "被动信号"}
    L2_MAP = {"pulse": "探测脉冲类", "communication": "通信类", "ship_noise": "舰船辐射噪声"}
    L3_MAP = {
        "CW": "CW连续波", "LFM": "LFM线性调频", "HFM": "HFM双曲调频",
        "2FSK": "2FSK二进制频移键控", "4FSK": "4FSK四进制频移键控",
        "BPSK": "BPSK二进制相移键控", "QPSK": "QPSK四进制相移键控",
        "OFDM": "OFDM正交频分复用",
        "cargo": "货船", "cruise": "邮轮", "fishing": "渔船",
        "warship": "军舰", "underwater_target": "水下目标",
    }
    L2_POOLS = {"active": ["pulse", "communication"], "passive": ["ship_noise"]}
    L3_BY_L2 = {
        "pulse": ["CW", "LFM", "HFM"],
        "communication": ["2FSK", "4FSK", "BPSK", "QPSK", "OFDM"],
        "ship_noise": ["cargo", "cruise", "fishing", "warship", "underwater_target"],
    }

    def __init__(self, mock_config: dict):
        self.accuracy = mock_config.get("accuracy", 0.85)
        self.l1_acc = mock_config.get("l1_accuracy", self.accuracy)
        self.l2_acc = mock_config.get("l2_accuracy", self.accuracy)
        self.l3_acc = mock_config.get("l3_accuracy", self.accuracy)
        self._rng = random.Random(42)

    def generate(self, audio_path: str, prompt: str,
                 max_new_tokens: int = 512, temperature: float = 0.1,
                 no_audio: bool = False) -> str:
        if no_audio:
            l1 = self._rng.choice(["active", "passive"])
            l2 = self._rng.choice(self.L2_POOLS[l1])
            l3 = self._random_l3(l2)
        else:
            l1, l2, l3 = self._pick_answer(audio_path)
            if self._rng.random() >= self.l1_acc:
                l1 = "passive" if l1 == "active" else "active"
                pool = self.L2_POOLS.get(l1, ["pulse"])
                l2 = self._rng.choice(pool)
                l3 = self._random_l3(l2)
            if self._rng.random() >= self.l2_acc:
                l2 = self._pick_wrong_l2(l1, l2)
                l3 = self._random_l3(l2)
            if self._rng.random() >= self.l3_acc:
                l3 = self._pick_wrong_l3(l2, l3)
        option_map = self._extract_options(prompt)
        target_letter = self._find_matching_option(option_map, l1, l2, l3)
        if target_letter:
            return f"{target_letter}"
        return self._natural_language_output(l1, l2, l3, prompt)

    def chat(self, audio_path: str, conversations: list,
             max_new_tokens: int = 512, temperature: float = 0.1,
             no_audio: bool = False) -> str:
        last_msg = ""
        for msg in reversed(conversations):
            if msg.get("from") == "human":
                last_msg = msg.get("value", "")
                break
        return self.generate(audio_path, last_msg, max_new_tokens,
                             temperature, no_audio=no_audio)

    def _extract_options(self, prompt: str) -> dict:
        import re
        option_map = {}
        for m in re.finditer(
            r'(?:^|\n)\s*([A-Za-z]+)\s*[\.\)\、\s]\s*(.+?)(?:\n|$)',
            prompt, re.MULTILINE,
        ):
            letter = m.group(1).strip()
            text = m.group(2).strip().rstrip('，,。.')
            if letter and text and len(letter) <= 2:
                option_map[letter.upper()] = text
        return option_map

    def _find_matching_option(self, option_map, l1, l2, l3):
        l1_name = self.L1_MAP.get(l1, l1)
        l2_name = self.L2_MAP.get(l2, l2)
        l3_name = self.L3_MAP.get(l3, l3)
        l1_c, l2_c, l3_c = set(), set(), set()
        for letter, text in option_map.items():
            if l1_name in text or l1 in text.lower():
                l1_c.add(letter)
            if l2_name in text or l2 in text.lower():
                l2_c.add(letter)
            if l3_name in text or l3.lower() in text.lower():
                l3_c.add(letter)
        if l3_c:
            for letter in l3_c:
                if letter in l2_c and letter in l1_c:
                    return letter
            return next(iter(l3_c))
        if l2_c:
            return next(iter(l2_c))
        if l1_c:
            return next(iter(l1_c))
        return ""

    def _natural_language_output(self, l1, l2, l3, prompt):
        if self._is_l1_question(prompt):
            return f"这是{self.L1_MAP[l1]}。"
        l2_name = self.L2_MAP[l2]
        l3_name = self.L3_MAP.get(l3, l3)
        return f"这是{self.L1_MAP[l1]}，属于{l2_name}中的{l3_name}。从时频特征来看，该信号具有典型的{l3_name}声学特征。"

    @staticmethod
    def _is_l1_question(prompt):
        has_active = "主动" in prompt or "发射" in prompt
        has_passive = "被动" in prompt or "接收" in prompt
        has_detail = "具体" in prompt or "子类" in prompt or "完整" in prompt or "细化" in prompt
        return has_active and has_passive and not has_detail

    def _pick_answer(self, audio_path):
        import os
        fname = os.path.basename(audio_path).lower()
        path_lower = audio_path.lower()
        if "05_ship" in path_lower:
            for l3 in self.L3_BY_L2["ship_noise"]:
                if l3 in path_lower:
                    return ("passive", "ship_noise", l3)
        if "pulsecom" in path_lower:
            for l2 in ("pulse", "communication"):
                for l3 in self.L3_BY_L2[l2]:
                    if l3.lower() in path_lower:
                        return ("active", l2, l3)
        for l3 in self.L3_BY_L2["ship_noise"]:
            if l3 in fname:
                return ("passive", "ship_noise", l3)
        for l2 in ("pulse", "communication"):
            for l3 in self.L3_BY_L2[l2]:
                if l3.lower() in fname:
                    return ("active", l2, l3)
        return ("active", "pulse", "CW")

    def _random_l3(self, l2):
        return self._rng.choice(self.L3_BY_L2.get(l2, ["CW"]))

    def _pick_wrong_l2(self, l1, correct_l2):
        pool = [s for s in self.L2_POOLS.get(l1, []) if s != correct_l2]
        return self._rng.choice(pool) if pool else correct_l2

    def _pick_wrong_l3(self, l2, correct_l3):
        pool = [s for s in self.L3_BY_L2.get(l2, []) if s != correct_l3]
        return self._rng.choice(pool) if pool else correct_l3


# ============================================================
# 注册后端 → ModelInference 调度表
# ============================================================
ModelInference.BACKENDS = {
    "mock":             MockModel,
    "qwen2_audio":      Qwen2AudioBackend,
    "aero1_audio":      Aero1AudioBackend,
    "voxtral_mini":     VoxtralMiniBackend,
    "af_next":          AudioFlamingoNextBackend,
    "kimi_audio":       KimiAudioBackend,
}
