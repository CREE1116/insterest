import asyncio
from google import genai
from app.core.config import settings
import json
import logging

logger = logging.getLogger(__name__)

class PromptOptimizer:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        if self.api_key:
            # 새로운 Google Gen AI SDK (v2) 클라이언트 초기화
            self.client = genai.Client(api_key=self.api_key)
            # 사용자가 명시한 최신 프리뷰 모델 명칭 적용
            self.model_name = "gemini-3.1-flash-lite-preview"
        else:
            self.client = None

    async def optimize(self, user_input: str) -> dict:
        """새로운 GenAI SDK 및 gemini-3.1-flash-lite-preview 모델을 사용하여 무드 해석"""
        fallback_result = {
            "image_prompt": f"A high-quality digital art capturing the mood of {user_input}, artistic style, cinematic lighting, 8k",
            "music_prompt": f"Ambient cinematic soundtrack matching {user_input}, lo-fi mood, instrumental",
            "mood": user_input,
            "suggested_title": user_input[:20] if user_input else "Untitled Mood"
        }

        if not self.client:
            return fallback_result

        prompt = f"""
        You are a creative director and expert prompt engineer. 
        The user will describe a specific "MOOD" or "ATMOSPHERE" in Korean or English. 
        Your task is to interpret this mood and generate two highly specific, synesthetic prompts in ENGLISH that perfectly capture it.

        User's Mood Description: "{user_input}"

        Respond ONLY with a JSON object in this format:
        {{
            "image_prompt": "string in English",
            "music_prompt": "string in English",
            "mood": "short summary in English",
            "suggested_title": "creative title in English"
        }}
        """
        
        try:
            logger.info(f"Interpreting mood via NEW SDK ({self.model_name}): {user_input}")
            # 새로운 SDK 방식: client.models.generate_content (Blocking call)
            # asyncio.to_thread를 사용하여 비동기 이벤트 루프를 방해하지 않음
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt
            )
            
            text = response.text.strip()
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].strip()
                
            result = json.loads(text)
            logger.info(f"Successfully generated prompts: {result.get('suggested_title')}")
            return result
        except Exception as e:
            logger.error(f"New SDK Prompting Error ({self.model_name}): {str(e)}")
            return fallback_result
