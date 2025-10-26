import logging
from typing import Iterator, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from src.config import Config

log = logging.getLogger(__name__)

class LLMEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        log.info(f"Loading model: {self.cfg.model_id} (precision={self.cfg.precision})")
        torch_dtype = None
        load_kwargs = {"device_map": self.cfg.device_map}

        if self.cfg.precision.lower() == "fp16":
            torch_dtype = torch.float16
        elif self.cfg.precision.lower() == "auto":
            torch_dtype = "auto"

        # 4-bit quant (Linux + bitsandbytes)
        if self.cfg.precision.lower() == "int4":
            load_kwargs.update(
                dict(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch_dtype,
            **load_kwargs,
        )
        log.info("Model loaded.")

    def _build_prompt(self, system: str, history: list[tuple[str, str]], user_msg: str) -> str:
        """
        Simple prompt format for instruct/chat models.
        For chat-tuned models that support chat templates, prefer apply_chat_template().
        """
        # If tokenizer supports chat templates, use them.
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "system", "content": system}]
            for u, a in history:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
            messages.append({"role": "user", "content": user_msg})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Fallback very simple format
        prompt = f"<system>\n{system}\n</system>\n"
        for u, a in history:
            prompt += f"<user>\n{u}\n</user>\n<assistant>\n{a}\n</assistant>\n"
        prompt += f"<user>\n{user_msg}\n</user>\n<assistant>\n"
        return prompt

    def generate_stream(
        self,
        system_prompt: str,
        history: list[tuple[str, str]],
        user_msg: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Token-by-token streaming generation using TextIteratorStreamer.
        """
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature = self.cfg.temperature if temperature is None else temperature
        top_p = self.cfg.top_p if top_p is None else top_p

        prompt = self._build_prompt(system_prompt, history, user_msg)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        # Kick off generation in background (we'll iterate streamer)
        import threading
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            yield text