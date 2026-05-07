import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def download_models():
    print("=== Model Download Start (MusicGen Only) ===")
    
    hf_home = os.environ.get("HF_HOME", "/app/models")
    print(f"Target Directory: {hf_home}")
    
    # 1. MusicGen model 만 다운로드 (Stable Diffusion 제거)
    print("\nDownloading MusicGen (facebook/musicgen-small)...")
    MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", cache_dir=hf_home)
    AutoProcessor.from_pretrained("facebook/musicgen-small", cache_dir=hf_home)
    
    print("\n=== MusicGen Model Downloaded Successfully ===")
    print("Stable Diffusion will be handled via Nano Banana API (External).")

if __name__ == "__main__":
    download_models()
