import sys
import types
import unittest

from testsite.core.inference import ModelInference, VoxtralBackend


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _InputIds:
    shape = (2, 2)


class _Inputs(dict):
    def __init__(self):
        super().__init__(input_ids=_InputIds())

    def to(self, device, dtype=None):
        self.device = device
        self.dtype = dtype
        return self


class _Generated:
    def __getitem__(self, key):
        if key != (slice(None), slice(2, None)):
            raise AssertionError(f"unexpected completion slice: {key}")
        return "completion-tokens"


class _Processor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, conversations):
        self.calls.append(conversations)
        return _Inputs()

    @staticmethod
    def batch_decode(values, skip_special_tokens):
        if values != "completion-tokens" or not skip_special_tokens:
            raise AssertionError("unexpected decode input")
        return [" one ", " two "]


class _Model:
    def generate(self, **kwargs):
        self.kwargs = kwargs
        return _Generated()


class VoxtralBackendTest(unittest.TestCase):
    def test_conversation_attaches_audio_only_to_first_user_turn(self):
        history = [
            {"from": "human", "value": "first"},
            {"from": "gpt", "value": "answer"},
            {"from": "human", "value": "follow-up"},
        ]

        conversation = VoxtralBackend._conversation(
            "sample.wav", history=history)

        self.assertEqual(
            conversation[0]["content"][0],
            {"type": "audio", "path": "sample.wav"},
        )
        self.assertEqual(
            conversation[1], {"role": "assistant", "content": "answer"})
        self.assertEqual(
            conversation[2]["content"],
            [{"type": "text", "text": "follow-up"}],
        )

    def test_generate_batch_uses_one_processor_and_model_call(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = _InferenceMode
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            backend = VoxtralBackend.__new__(VoxtralBackend)
            backend.processor = _Processor()
            backend.model = _Model()
            backend.device = "cuda:0"
            backend.torch_dtype = "bfloat16"
            backend.default_temperature = 0.0
            backend.max_new_tokens = 256

            outputs = backend.generate_batch(
                ["a.wav", "b.wav"], ["A", "B"], max_new_tokens=32)

            self.assertEqual(outputs, ["one", "two"])
            self.assertEqual(len(backend.processor.calls), 1)
            self.assertEqual(len(backend.processor.calls[0]), 2)
            self.assertEqual(backend.model.kwargs["max_new_tokens"], 32)
            self.assertFalse(backend.model.kwargs["do_sample"])
        finally:
            if previous_torch is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = previous_torch

    def test_wrapper_honors_backend_generation_defaults(self):
        backend = VoxtralBackend.__new__(VoxtralBackend)
        backend.max_new_tokens = 256
        backend.default_temperature = 0.0
        seen = []

        def fake_infer(_self, _audio, _prompt, _history,
                       max_new_tokens, temperature, _no_audio):
            seen.append((max_new_tokens, temperature))
            return "ok"

        backend._infer = types.MethodType(fake_infer, backend)
        wrapper = ModelInference.__new__(ModelInference)
        wrapper._model = backend

        self.assertEqual(wrapper.generate("a.wav", "prompt"), "ok")
        self.assertEqual(seen, [(256, 0.0)])


if __name__ == "__main__":
    unittest.main()
