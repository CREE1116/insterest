import asyncio
import json
import logging
import os
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from aiokafka import AIOKafkaProducer

from app.services.prompt_service import PromptOptimizer
from app.core.config import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME)

# Kafka Producer 전역 변수
producer = None

@app.on_event("startup")
async def startup_event():
    global producer
    logger.info(f"Connecting to Kafka at {settings.KAFKA_BOOTSTRAP_SERVERS}...")
    try:
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await producer.start()
        logger.info("Kafka Producer started successfully.")
    except Exception as e:
        logger.error(f"Failed to start Kafka Producer: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    global producer
    if producer:
        await producer.stop()
        logger.info("Kafka Producer stopped.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

prompt_optimizer = PromptOptimizer()
_content_generator = None
_load_lock = threading.Lock()
generation_lock = asyncio.Lock()

def load_generator_sync():
    global _content_generator
    with _load_lock:
        if _content_generator is None:
            from app.services.generation_service import ContentGenerator
            logger.info("Initializing ContentGenerator (Global Singleton)...")
            _content_generator = ContentGenerator()
            logger.info("ContentGenerator ready.")

async def heartbeat(websocket: WebSocket):
    try:
        while True:
            await asyncio.sleep(15)
            await websocket.send_json({"status": "heartbeat", "message": "💓 서버와 연결 유지 중..."})
    except: pass

from jose import jwt, JWTError

@app.websocket("/api/v1/ws/generate")
async def websocket_generate(websocket: WebSocket):
    # 쿠키에서 access_token 확인
    token = websocket.cookies.get("access_token")
    if not token:
        await websocket.accept()
        await websocket.send_json({"status": "error", "message": "인증이 필요합니다. 로그인 후 이용해주세요."})
        await websocket.close()
        return

    try:
        # 토큰 검증 및 유저 ID 추출
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise JWTError("User ID not found in token")
    except JWTError:
        await websocket.accept()
        await websocket.send_json({"status": "error", "message": "유효하지 않은 토큰입니다."})
        await websocket.close()
        return

    await websocket.accept()
    logger.info(f"New authenticated WebSocket connection accepted for user: {user_id}")
    
    # 1. 사용자의 요청 데이터를 먼저 받습니다.
    try:
        data = await websocket.receive_text()
        request = json.loads(data)
        user_input = request.get("user_input")
    except Exception as e:
        logger.error(f"Failed to receive initial data: {e}")
        await websocket.close()
        return

    hb_task = asyncio.create_task(heartbeat(websocket))
    global _content_generator
    
    try:
        # 2. 모델 로딩이 필요 없으므로 바로 준비 완료 상태를 보냅니다.
        if _content_generator is None:
            load_generator_sync()
        
        await websocket.send_json({"status": "ready", "message": "✅ 준비 완료!", "progress": 25})

        # 3. 이미 데이터를 받았으므로 바로 생성을 시작합니다.
        async with generation_lock:
            await websocket.send_json({"status": "analyzing", "message": "🤖 무드 해석 중...", "progress": 40})
            optimized = await prompt_optimizer.optimize(user_input)
            
            include_music = request.get("include_music", True)
            status_msg = "🎨 이미지와 🎵 음악 생성 중..." if include_music else "🎨 이미지 생성 중..."
            await websocket.send_json({"status": "generating", "message": status_msg, "progress": 60})
            
            # 병렬 실행 및 예외 처리
            include_music = request.get("include_music", True)
            
            image_task = _content_generator.generate_image(optimized.get("image_prompt", user_input))
            music_task = None
            if include_music:
                music_task = _content_generator.generate_music(optimized.get("music_prompt", user_input), request.get("duration", 10))
            
            # 생성 진행 중 웹소켓 연결 유지를 위해 주기적으로 업데이트 전송
            async def progress_heartbeat():
                while not (image_task.done() and (music_task is None or music_task.done())):
                    try:
                        await asyncio.sleep(5)
                        if not websocket.client_state.name == "CONNECTED": break
                        await websocket.send_json({"status": "generating", "message": "🎵 미디어 생성 중... (잠시만 기다려 주세요)", "progress": 70})
                    except: break

            heartbeat_task = asyncio.create_task(progress_heartbeat())
            
            try:
                tasks = [image_task]
                if music_task:
                    tasks.append(music_task)
                
                results = await asyncio.gather(*tasks)
                image_path = results[0]
                music_path = results[1] if len(results) > 1 else None
            finally:
                heartbeat_task.cancel()

            # Kafka로 생성 완료 메시지 전송
            global producer
            if producer:
                kafka_msg = {
                    "user_id": user_id, # 토큰에서 추출한 실제 유저 ID 사용
                    "title": optimized.get("suggested_title", "AI Content"),
                    "mood": optimized.get("mood", "Unique Atmosphere"),
                    "image_url": image_path,
                    "music_url": music_path,
                    "image_prompt": optimized.get("image_prompt", "Default Image Prompt"),
                    "music_prompt": optimized.get("music_prompt", "Default Music Prompt") if include_music else None,
                    "content_type": "PHOTO_SOUND" if include_music else "PHOTO"
                }
                try:
                    await producer.send_and_wait("generation.completed", kafka_msg)
                    logger.info(f"Sent completion message to Kafka for: {kafka_msg['title']}")
                except Exception as e:
                    logger.error(f"Failed to send Kafka message: {e}")

            # 확실하게 데이터를 구성하여 전송
            response_data = {
                "status": "completed",
                "message": "🎉 생성 완료!",
                "progress": 100,
                "data": {
                    "content_id": optimized.get("content_id", ""), 
                    "title": optimized.get("suggested_title", "AI Content"),
                    "mood": optimized.get("mood", "Unique Atmosphere"),
                    "image_url": image_path,
                    "audio_url": music_path, # Explicitly named for frontend
                    "music_url": music_path,
                    "image_prompt": optimized.get("image_prompt", "Default Image Prompt"),
                    "music_prompt": optimized.get("music_prompt", "Default Music Prompt") if include_music else None
                }
            }
            logger.info(f"Sending final response: {optimized.get('suggested_title')}")
            await websocket.send_json(response_data)

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WS Error: {e}")
        try: await websocket.send_json({"status": "error", "message": str(e)})
        except: pass
    finally:
        hb_task.cancel()
        try: await websocket.close()
        except: pass
