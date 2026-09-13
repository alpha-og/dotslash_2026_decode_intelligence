import modal

app = modal.App("gpt2-demo")

# CPU-only image for free tier. Use distilgpt2 (82M) - fastest cold start.
# Swap to "gpt2" (124M) or "gpt2-medium" if you want more quality (still CPU ok, slower).
MODEL_NAME = "distilgpt2"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "transformers==4.44.2",
    "torch==2.4.1",
    "accelerate==0.34.2",
    "huggingface_hub==0.24.7",
    "fastapi[standard]==0.115.0",
)

# Persist HF cache so model download happens once (saves time & bandwidth)
volume = modal.Volume.from_name("hf-cache-demo", create_if_missing=True)
CACHE_DIR = "/root/.cache/huggingface"


@app.cls(
    image=image,
    volumes={CACHE_DIR: volume},
    scaledown_window=300,  # keep warm 5 min for demo
)
class GPT2Model:
    @modal.enter()
    def load(self):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        print(f"Loading model: {MODEL_NAME}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        # GPT2 has no pad token, set to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        print("Model loaded on CPU")

    @modal.method()
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7):
        import torch

        if not prompt or not prompt.strip():
            prompt = "Hello, I am"

        # Clamp params for free-tier safety
        max_new_tokens = max(1, min(int(max_new_tokens), 100))
        temperature = max(0.1, min(float(temperature), 2.0))

        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens, not the prompt
        text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return {"prompt": prompt, "generated_text": text.strip(), "model": MODEL_NAME}


# --- HTTPS Endpoints ---


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": "cpu",
        "message": "POST to /generate with {'prompt': '...'} ",
    }


@app.function(image=image, volumes={CACHE_DIR: volume})
@modal.fastapi_endpoint(method="POST")
def generate(item: dict):
    """
    Main inference endpoint.
    POST https://<app>--generate.modal.run
    Body: {"prompt": "Once upon a time", "max_new_tokens": 50, "temperature": 0.7}
    """
    prompt = item.get("prompt", "") if isinstance(item, dict) else ""
    max_new_tokens = item.get("max_new_tokens", 50) if isinstance(item, dict) else 50
    temperature = item.get("temperature", 0.7) if isinstance(item, dict) else 0.7

    # Call the class method (runs in separate container with model loaded)
    result = GPT2Model().generate.remote(prompt, max_new_tokens, temperature)
    return result


@app.local_entrypoint()
def main():
    prompt = "Hello, I am a language model"
    print(f"Testing Modal generation with prompt: {prompt}")
    result = GPT2Model().generate.remote(prompt, max_new_tokens=30)
    print(result)
