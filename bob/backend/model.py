# model.py

from transformers import pipeline

print("[TinyLlama INIT] Loading model...")

llm_pipeline = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_kwargs={"torch_dtype": "auto"},
    device_map="auto"
)

print("[TinyLlama INIT] Model is ready.")

def get_llm_pipeline():
    return llm_pipeline
