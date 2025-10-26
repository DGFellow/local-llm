import os
from dataclasses import dataclass

@dataclass
class Config:
    # Change these to taste (or via .env)
    model_id: str = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    # Options: "auto", "fp16", "int4" (int4 needs bitsandbytes, usually Linux)
    precision: str = os.getenv("PRECISION", "auto")
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    device_map: str = os.getenv("DEVICE_MAP", "auto")   # "auto" spreads across GPU/CPU as needed
    chat_system_prompt: str = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant.")

def save_env(model_id: str, precision: str):
    path = os.path.join(os.getcwd(), ".env")
    # Minimal: keep only MODEL_ID and PRECISION lines fresh; append/replace if present.
    lines = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    def upsert(key: str, value: str):
        kv = f"{key}="
        for i, line in enumerate(lines):
            if line.startswith(kv):
                lines[i] = f"{key}={value}"
                break
        else:
            # runs only if the loop didn't break (no existing key found)
            lines.append(f"{key}={value}")

    # Make/update both keys
    upsert("MODEL_ID", model_id)
    upsert("PRECISION", precision)

    # Write back (ensure LF newlines)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")