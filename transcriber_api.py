import os
import io
import base64
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoConfig
import onnxruntime
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Load models

## Load config and processor
from huggingface_hub import login

# Get HuggingFace token from environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("WARNING: HUGGINGFACE_TOKEN not found in .env file")
    print("Please add your HuggingFace token to the .env file")
    print("You can get a token from: https://huggingface.co/settings/tokens")
else:
    login(token=hf_token)
    print("Successfully logged in to HuggingFace")


model_id = "google/gemma-3n-E2B-it"
# Download the processor and configuration files while saving them in the cache directory
processor = AutoProcessor.from_pretrained(model_id, cache_dir="onnx")
config = AutoConfig.from_pretrained(model_id, cache_dir="onnx")


# === Configuration ===
model_id = "google/gemma-3n-E2B-it"
cache_dir = "onnx"
model_dir = "gemma-3n-E2B-it-ONNX"

# Load processor and config
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)

# Load ONNX model paths
embed_model_path = os.path.join(model_dir, "onnx/embed_tokens_uint8.onnx")
vision_model_path = os.path.join(model_dir, "onnx/vision_encoder_quantized.onnx")
decoder_model_path = os.path.join(model_dir, "onnx/decoder_model_merged_q4f16.onnx")

# Load ONNX sessions
providers = ["CPUExecutionProvider"]  # or CUDAExecutionProvider if available
embed_session = onnxruntime.InferenceSession(embed_model_path, providers=providers)
print(f"Loaded embed session from {embed_model_path}")
vision_session = onnxruntime.InferenceSession(vision_model_path, providers=providers)
print(f"Loaded vision session from {vision_model_path}")
decoder_session = onnxruntime.InferenceSession(decoder_model_path, providers=providers)
print(f"Loaded decoder session from {decoder_model_path}")

# === Config values ===
num_key_value_heads = config.text_config.num_key_value_heads
head_dim = config.text_config.head_dim
num_hidden_layers = config.text_config.num_hidden_layers
eos_token_id = 106
image_token_id = config.image_token_id
audio_token_id = config.audio_token_id

# === FastAPI app ===
app = FastAPI()


class SlideRequest(BaseModel):
    image_base64: str


@app.post("/transcribe-slide")
async def transcribe_slide(request: SlideRequest):
    try:
        # Decode image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Prepare message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "In detail, describe the following image.",
                    },
                    {"type": "image", "image": image},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].numpy()
        attention_mask = inputs["attention_mask"].numpy()
        position_ids = np.cumsum(attention_mask, axis=-1) - 1
        pixel_values = (
            inputs["pixel_values"].numpy() if "pixel_values" in inputs else None
        )
        input_features = inputs.get("input_features", None)
        input_features_mask = inputs.get("input_features_mask", None)

        if input_features is not None:
            input_features = input_features.numpy().astype(np.float32)
        if input_features_mask is not None:
            input_features_mask = input_features_mask.numpy()

        # Prepare decoder inputs
        batch_size = input_ids.shape[0]
        past_key_values = {
            f"past_key_values.{layer}.{kv}": np.zeros(
                [batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32
            )
            for layer in range(num_hidden_layers)
            for kv in ("key", "value")
        }

        max_new_tokens = 512
        generated_tokens = np.array([[]], dtype=np.int64)
        image_features = None
        audio_features = None

        for _ in range(max_new_tokens):
            inputs_embeds, per_layer_inputs = embed_session.run(
                None, {"input_ids": input_ids}
            )

            if image_features is None and pixel_values is not None:
                image_features = vision_session.run(
                    ["image_features"], {"pixel_values": pixel_values}
                )[0]
                mask = (input_ids == image_token_id).reshape(-1)
                flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                flat_embeds[mask] = image_features.reshape(-1, image_features.shape[-1])
                inputs_embeds = flat_embeds.reshape(inputs_embeds.shape)

            logits, *present_key_values = decoder_session.run(
                None,
                dict(
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    position_ids=position_ids,
                    **past_key_values,
                ),
            )

            input_ids = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1

            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

            generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
            if (input_ids == eos_token_id).all():
                break

        generated_text = processor.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]
        return {"script": generated_text.strip()}

    except Exception as e:
        return {"error": str(e)}