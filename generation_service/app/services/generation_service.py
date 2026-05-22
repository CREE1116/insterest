import asyncio
import aiofiles
import os
import uuid
import logging
import httpx
from urllib.parse import quote
from app.core.config import settings

logger = logging.getLogger(__name__)

class ContentGenerator:
    def __init__(self):
        self.output_dir = settings.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Pollinations API Key (Images & Music)
        self.pollinations_api_key = "sk_OrNrk999cQnTnqu16sepfu54uwCXRBPx"
        self.pollinations_host = "gen.pollinations.ai"

    async def generate_image(self, prompt: str) -> str:
        """Pollinations Flux API를 사용하여 이미지 생성"""
        clean_prompt = prompt.replace("\n", " ").strip()
        if len(clean_prompt) > 1000:
            clean_prompt = clean_prompt[:1000]
            
        filename = f"v15_{uuid.uuid4()}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        encoded_prompt = quote(clean_prompt)
        url = f"https://{self.pollinations_host}/image/{encoded_prompt}?width=1024&height=1024&nologo=true&model=flux"
        
        headers = {
            "Authorization": f"Bearer {self.pollinations_api_key}"
        }
        
        logger.info(f"🎨 Generating Image: {clean_prompt[:50]}...")
        
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                res = await client.get(url, headers=headers)
                
                content_type = res.headers.get("content-type", "").lower()
                is_html = b"<!DOCTYPE" in res.content[:100] or b"<html" in res.content[:100]
                
                if res.status_code == 200 and "image" in content_type and not is_html:
                    async with aiofiles.open(filepath, "wb") as f:
                        await f.write(res.content)
                    return f"/outputs/{filename}"
                else:
                    raise Exception(f"Image generation failed: {res.status_code}")
                    
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            raise e

    async def generate_music(self, prompt: str, duration_sec: int = 10) -> str:
        """Pollinations Acestop API를 사용하여 음악 생성 (facebook/musicgen-small 대안)"""
        clean_music_prompt = prompt.replace("\n", " ").strip()[:500]
        logger.info(f"🎵 Generating Music via Pollinations (acestep): {clean_music_prompt[:50]}...")
        
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(self.output_dir, filename)
        
        encoded_prompt = quote(clean_music_prompt)
        # acestep 모델은 오픈소스로 무료 사용 가능하며 성능이 뛰어납니다.
        url = f"https://{self.pollinations_host}/audio/{encoded_prompt}?model=acestep"
        
        headers = {
            "Authorization": f"Bearer {self.pollinations_api_key}"
        }
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    async with aiofiles.open(filepath, "wb") as f:
                        await f.write(response.content)
                    logger.info(f"✅ Music generated successfully: {filename}")
                    return f"/outputs/{filename}"
                else:
                    error_msg = f"Pollinations Audio API Error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            logger.error(f"Music generation error: {str(e)}")
            raise e
