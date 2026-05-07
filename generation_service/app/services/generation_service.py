import asyncio
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile as wavfile
import numpy as np
import os
import uuid
import logging
import httpx
from urllib.parse import quote
from app.core.config import settings

logger = logging.getLogger(__name__)

OFFLINE = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

class ContentGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = settings.OUTPUT_DIR
        self.hf_home = settings.HF_HOME
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.hf_home, exist_ok=True)
        
        # Pollinations API Key
        self.pollinations_api_key = "sk_OrNrk999cQnTnqu16sepfu54uwCXRBPx"
        
        self.image_model = None
        self.music_model = None
        self.music_processor = None

    async def _load_music_model(self):
        if self.music_model is None:
            model_id = "facebook/musicgen-small"
            logger.info(f"Loading Official model: {model_id} (Offline: {OFFLINE}, Cache: {self.hf_home})...")
            
            # CPU/GPU Optimization: bfloat16 is not supported on all CPUs, so use float32 as a safe fallback.
            if self.device == "cpu":
                dtype = torch.float32
            else:
                dtype = torch.float16
            
            try:
                model = await asyncio.to_thread(
                    MusicgenForConditionalGeneration.from_pretrained,
                    model_id, 
                    cache_dir=self.hf_home,
                    local_files_only=OFFLINE,
                    torch_dtype=dtype,
                    attn_implementation="eager" # Fixed: T5 doesn't support sdpa yet
                )
                self.music_model = model.to(self.device)
            except Exception as e:
                logger.error(f"Failed to load music model with {dtype}: {e}")
                if dtype != torch.float32:
                    logger.info("Retrying with float32...")
                    model = await asyncio.to_thread(
                        MusicgenForConditionalGeneration.from_pretrained,
                        model_id, 
                        cache_dir=self.hf_home,
                        local_files_only=OFFLINE,
                        torch_dtype=torch.float32,
                        attn_implementation="eager"
                    )
                    self.music_model = model.to(self.device)
                else:
                    raise e
            
            self.music_processor = await asyncio.to_thread(
                AutoProcessor.from_pretrained,
                model_id,
                cache_dir=self.hf_home,
                local_files_only=OFFLINE
            )

    async def generate_image(self, prompt: str) -> str:
        """제공된 API 키와 엔드포인트를 사용하여 Pollinations Flux 이미지 생성"""
        clean_prompt = prompt.replace("\n", " ").strip()
        if len(clean_prompt) > 1000:
            clean_prompt = clean_prompt[:1000]
            
        filename = f"v15_{uuid.uuid4()}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        # 공식 문서 엔드포인트: /image/{prompt}
        host = "gen.pollinations.ai"
        encoded_prompt = quote(clean_prompt)
        url = f"https://{host}/image/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"
        
        headers = {
            "Authorization": f"Bearer {self.pollinations_api_key}"
        }
        
        logger.info(f"==== GENERATING IMAGE V14 ====")
        logger.info(f"URL: {url}")
        
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                res = await client.get(url, headers=headers)
                
                content_type = res.headers.get("content-type", "").lower()
                
                # 데이터가 HTML인지 더 꼼꼼히 체크
                is_html = b"<!DOCTYPE" in res.content[:100] or b"<html" in res.content[:100]
                
                if res.status_code == 200 and "image" in content_type and not is_html:
                    with open(filepath, "wb") as f:
                        f.write(res.content)
                    logger.info(f"==== V14 SUCCESS: {filename} ({len(res.content)} bytes) ====")
                    return f"/outputs/{filename}"
                else:
                    msg = f"V14 REJECTED: Status={res.status_code}, Type={content_type}, IsHTML={is_html}"
                    logger.error(msg)
                    raise Exception(msg)
                    
        except Exception as e:
            logger.error(f"Image generation failed: {str(e)}")
            raise e

    async def generate_music(self, prompt: str, duration_sec: int = 8) -> str:
        await self._load_music_model()
        clean_music_prompt = prompt.replace("\n", " ").strip()[:500]
        logger.info(f"Generating music for: {clean_music_prompt}")
        
        # 입력 전처리
        inputs = self.music_processor(text=[clean_music_prompt], padding=True, return_tensors="pt").to(self.device)
        
        # 최적화된 생성 파라미터 적용 (8초 분량, 샘플링 방식)
        audio_values = await asyncio.to_thread(
            self.music_model.generate,
            **inputs, 
            do_sample=True,
            guidance_scale=3.0,
            max_new_tokens=int(duration_sec * 50) # 8초 -> 400 tokens
        )
        
        sampling_rate = self.music_model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy()
        
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(self.output_dir, filename)
        wavfile.write(filepath, rate=sampling_rate, data=audio_data)
        return f"/outputs/{filename}"
