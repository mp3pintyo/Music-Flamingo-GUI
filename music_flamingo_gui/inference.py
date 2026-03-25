from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Thread
from typing import Any

import torch

from music_flamingo_gui.formatting import split_reasoning

try:
    from transformers import AutoProcessor, BitsAndBytesConfig, MusicFlamingoForConditionalGeneration
except ImportError as exc:  # pragma: no cover - surfaced at runtime for missing deps
    AutoProcessor = None
    BitsAndBytesConfig = None
    MusicFlamingoForConditionalGeneration = None
    TRANSFORMERS_IMPORT_ERROR = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None


DEFAULT_MODEL_ID = "nvidia/music-flamingo-think-2601-hf"


@dataclass(frozen=True)
class ModelOptions:
    model_id: str = DEFAULT_MODEL_ID
    quantization: str = "4bit"
    attn_implementation: str = "sdpa"
    gpu_memory_limit_gib: int = 14
    cpu_offload: bool = True


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 768
    temperature: float = 0.6
    top_p: float = 0.9


@dataclass
class GenerationResult:
    conversation: list[dict[str, Any]]
    raw_text: str
    reasoning: str
    final_answer: str
    status_note: str


class MusicFlamingoService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._model = None
        self._processor = None
        self._options: ModelOptions | None = None
        self._compute_dtype = torch.float32

    def load_model(self, options: ModelOptions) -> str:
        self._ensure_transformers_ready()

        with self._lock:
            if self._model is not None and self._options == options:
                return self._describe_runtime(options)

            self._unload_locked()

            self._compute_dtype = self._pick_compute_dtype()
            load_kwargs: dict[str, Any] = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self._compute_dtype,
                "attn_implementation": options.attn_implementation,
            }

            if torch.cuda.is_available():
                load_kwargs["device_map"] = "auto"
                max_memory = self._build_max_memory(options)
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
                    offload_folder = Path.cwd() / "hf-offload"
                    offload_folder.mkdir(parents=True, exist_ok=True)
                    load_kwargs["offload_folder"] = str(offload_folder)
                    load_kwargs["offload_state_dict"] = True

            quantization_config = self._build_quantization_config(options)
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config

            self._processor = AutoProcessor.from_pretrained(options.model_id)
            self._model = MusicFlamingoForConditionalGeneration.from_pretrained(options.model_id, **load_kwargs)
            self._options = options

            return self._describe_runtime(options)

    def unload_model(self) -> str:
        with self._lock:
            self._unload_locked()
        return "A modell ki lett pakolva a memoriabol."

    def generate_reply(
        self,
        prompt: str,
        audio_path: str | None,
        history: list[dict[str, Any]] | None,
        model_options: ModelOptions,
        generation: GenerationOptions,
    ) -> GenerationResult:
        if not prompt.strip() and not audio_path:
            raise ValueError("Adj meg szoveget vagy tolts fel egy hangfajlt.")

        self.load_model(model_options)

        with self._lock:
            conversation = copy.deepcopy(history or [])
            user_content: list[dict[str, str]] = []

            if prompt.strip():
                user_content.append({"type": "text", "text": prompt.strip()})

            if audio_path:
                self._ensure_audio_backend_ready()
                user_content.append({"type": "audio", "path": audio_path})

            conversation.append({"role": "user", "content": user_content})

            inputs = self._processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            inputs = self._move_inputs(inputs)

            prompt_length = int(inputs["input_ids"].shape[1])
            effective_max_new_tokens, status_note = self._resolve_generation_budget(prompt_length, generation.max_new_tokens)

            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(self._compute_dtype)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": effective_max_new_tokens,
                "max_length": prompt_length + effective_max_new_tokens,
            }
            if generation.temperature > 0:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": generation.temperature,
                        "top_p": generation.top_p,
                    }
                )
            else:
                generation_kwargs["do_sample"] = False

            with torch.inference_mode():
                outputs = self._model.generate(**inputs, **generation_kwargs)

            generated_tokens = outputs[:, prompt_length:]
            raw_text = self._processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
            reasoning, final_answer = split_reasoning(raw_text)

            conversation.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": raw_text}],
                }
            )

            return GenerationResult(
                conversation=conversation,
                raw_text=raw_text,
                reasoning=reasoning,
                final_answer=final_answer or raw_text,
                status_note=status_note,
            )

    def generate_stream(
        self,
        prompt: str,
        audio_path: str | None,
        history: list[dict[str, Any]] | None,
        model_options: ModelOptions,
        generation: GenerationOptions,
    ):
        """Generátor: intermediate és final update dict-eket yield-el."""
        from transformers import TextIteratorStreamer

        if not (prompt or "").strip() and not audio_path:
            raise ValueError("Adj meg szoveget vagy tolts fel egy hangfajlt.")

        self.load_model(model_options)

        with self._lock:
            conversation = copy.deepcopy(history or [])
            user_content: list[dict[str, str]] = []
            if (prompt or "").strip():
                user_content.append({"type": "text", "text": prompt.strip()})
            if audio_path:
                self._ensure_audio_backend_ready()
                user_content.append({"type": "audio", "path": audio_path})
            conversation.append({"role": "user", "content": user_content})

            inputs = self._processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            inputs = self._move_inputs(inputs)

            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(self._compute_dtype)

            prompt_length = int(inputs["input_ids"].shape[1])
            effective_max_new_tokens, status_note = self._resolve_generation_budget(
                prompt_length, generation.max_new_tokens
            )

            tokenizer = getattr(self._processor, "tokenizer", None) or self._processor
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=300.0
            )

            gen_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": effective_max_new_tokens,
                "max_length": prompt_length + effective_max_new_tokens,
                "streamer": streamer,
            }
            if generation.temperature > 0:
                gen_kwargs.update(
                    {"do_sample": True, "temperature": generation.temperature, "top_p": generation.top_p}
                )
            else:
                gen_kwargs["do_sample"] = False

            model_ref = self._model

        errors: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    model_ref.generate(**gen_kwargs)
            except Exception as exc:
                errors.append(exc)
                try:
                    streamer.text_queue.put(streamer.stop_signal)
                except Exception:
                    pass

        thread = Thread(target=_run, daemon=True)
        thread.start()

        full_text = ""
        try:
            for chunk in streamer:
                full_text += chunk
                if full_text.strip():
                    reasoning, answer = split_reasoning(full_text)
                    yield {
                        "done": False,
                        "full_text": full_text,
                        "reasoning": reasoning,
                        "answer": answer or full_text,
                    }
        finally:
            thread.join(timeout=10)

        if errors:
            raise errors[0]

        if not full_text.strip():
            raise RuntimeError(
                "A modell ures valaszt generalt. Lehet, hogy az audio tul hosszu, "
                "vagy a kontextus tul rovid. Roviditsd a bemenetet."
            )

        reasoning, final_answer = split_reasoning(full_text)
        final_conv = conversation + [{"role": "assistant", "content": [{"type": "text", "text": full_text}]}]

        yield {
            "done": True,
            "full_text": full_text,
            "reasoning": reasoning,
            "answer": final_answer or full_text,
            "conversation": final_conv,
            "status_note": status_note,
        }

    def _ensure_transformers_ready(self) -> None:
        if TRANSFORMERS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Nem sikerult importalni a Music Flamingo-hoz szukseges Transformers buildet. "
                "Telepitsd a requirements.txt tartalmat, kulonosen a modular-mf agrol a transformers csomagot."
            ) from TRANSFORMERS_IMPORT_ERROR

    def _ensure_audio_backend_ready(self) -> None:
        try:
            import librosa  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Az audio feldolgozashoz hianyzik a librosa csomag. Telepitsd a requirements.txt tartalmat, vagy futtasd: pip install librosa"
            ) from exc

    def _build_quantization_config(self, options: ModelOptions) -> BitsAndBytesConfig | None:
        if options.quantization != "4bit":
            return None

        if BitsAndBytesConfig is None:
            raise RuntimeError("A 4 bites betolteshez a bitsandbytes tamogatas nem erheto el.")

        if not torch.cuda.is_available():
            raise RuntimeError("A 4 bites modhoz CUDA-s NVIDIA GPU kell.")

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=self._compute_dtype,
        )

    def _build_max_memory(self, options: ModelOptions) -> dict[Any, str] | None:
        if not torch.cuda.is_available():
            return None

        gpu_cap = max(8, min(options.gpu_memory_limit_gib, 15))
        max_memory: dict[Any, str] = {0: f"{gpu_cap}GiB"}
        if options.cpu_offload:
            max_memory["cpu"] = "48GiB"
        return max_memory

    def _pick_compute_dtype(self) -> torch.dtype:
        if not torch.cuda.is_available():
            return torch.float32

        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        device = self._input_device()
        moved: dict[str, Any] = {}

        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(device)
            else:
                moved[key] = value

        return moved

    def _resolve_generation_budget(self, prompt_length: int, requested_new_tokens: int) -> tuple[int, str]:
        if self._model is None:
            raise RuntimeError("A modell nincs betoltve.")

        # A szovegmodell limitjet hasznaljuk (32768), NEM az audio-adapter positional limitjet (1200).
        text_config = getattr(self._model.config, "text_config", None)
        if text_config is not None:
            context_limit = (
                getattr(text_config, "max_position_embeddings", None)
                or getattr(text_config, "model_max_length", None)
            )
        else:
            context_limit = getattr(self._model.config, "max_position_embeddings", None)

        # Ha a kiolvasott limit az audio-adapter 1200-as erteke lenne, ne korlátozzuk a generalast.
        if not context_limit or int(context_limit) < 2048:
            return requested_new_tokens, "Valasz elkeszult."

        remaining_tokens = int(context_limit) - prompt_length
        if remaining_tokens <= 32:
            raise RuntimeError(
                "A bemenet tul hosszu. Roviditsd a promptot, hasznalj rovidebb audioreszletet, vagy torold a korabbi beszelgetest."
            )

        effective_max_new_tokens = max(32, min(requested_new_tokens, remaining_tokens))
        if effective_max_new_tokens < requested_new_tokens:
            return (
                effective_max_new_tokens,
                f"A valasz elkeszult. Hossz automatikusan {effective_max_new_tokens} tokenre csokkentve (kontextushatar: {context_limit} token).",
            )

        return effective_max_new_tokens, "Valasz elkeszult."

    def _input_device(self) -> torch.device:
        if self._model is None:
            raise RuntimeError("A modell nincs betoltve.")

        model_device = getattr(self._model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return torch.device(model_device)

        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for mapped_device in device_map.values():
                if isinstance(mapped_device, int):
                    return torch.device(f"cuda:{mapped_device}")
                if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk"}:
                    return torch.device(mapped_device)

        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _describe_runtime(self, options: ModelOptions) -> str:
        precision = "4-bit NF4" if options.quantization == "4bit" else str(self._compute_dtype).replace("torch.", "")
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        offload = f" GPU limit: {options.gpu_memory_limit_gib} GiB." if torch.cuda.is_available() else ""
        cpu_offload = " CPU offload aktiv." if options.cpu_offload and torch.cuda.is_available() else ""
        return f"A modell keszen all. Mod: {precision}. Backend: {device}.{offload}{cpu_offload}"

    def _unload_locked(self) -> None:
        self._model = None
        self._processor = None
        self._options = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_SERVICE = MusicFlamingoService()


def get_service() -> MusicFlamingoService:
    return _SERVICE
