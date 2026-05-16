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
        You are a world-class Creative Director specializing in Synesthesia and Multi-modal AI generation.
        Your goal is to interpret the user's "MOOD" and translate it into professional-grade AI prompts.

        ### CRITICAL RULE:
        - **PRESERVE THE CORE SUBJECT**: If the user mentions a "Cat", the main subject MUST be a Cat. Do NOT replace it with unrelated elements like humans or different animals.
        - **BE ARTISTIC & MOODY**: You don't need to be realistic. Focus on the *aesthetic*, *vibe*, and *synesthetic harmony*. Create an artistic interpretation that makes the user "feel" the mood.

        ### Guidelines for Image Prompt:
        - Style: Artistic, atmospheric, visually striking. (Can be digital art, painting, cinematic, or any style that fits the mood).
        - Elements: Focus on the user's subject. Enhance it with poetic lighting, symbolic colors, and creative compositions.

        ### Guidelines for Music Prompt:
        - Style: Atmospheric, instrumental, high-quality audio.
        - Elements: Specify genres, instruments, tempo, and emotional resonance matching the subject.

        ### Few-shot Examples:
        User Mood: "귀여운 고양이가 있는 일상" (Daily life with a cute cat)
        Result: {{
            "image_prompt": "A fluffy ginger cat napping on a sun-drenched wooden windowsill, soft dust motes dancing in golden sunlight, cozy living room atmosphere, ultra-detailed fur texture, macro shot, cinematic lighting, warm and peaceful mood, 8k.",
            "music_prompt": "Gentle acoustic guitar melody, soft shaker percussion, light and playful piano notes, medium-slow tempo, heartwarming and relaxing daytime mood, high-fidelity.",
            "mood": "Peaceful afternoon with a cat",
            "suggested_title": "Golden Nap Time"
        }}

        User's Mood Description: "{user_input}"

        Respond ONLY with a JSON object in this format:
        {{
            "image_prompt": "detailed English prompt",
            "music_prompt": "detailed English prompt",
            "mood": "concise English summary",
            "suggested_title": "creative English title"
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
