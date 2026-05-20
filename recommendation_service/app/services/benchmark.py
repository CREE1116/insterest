import io
import json
import zipfile
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import httpx
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple
from sqlalchemy import text
from app.ml.intelligence import nlp_embedder
from app.core.logging import logger

class BenchmarkService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_synthetic_image(self, shape: str, color: str) -> bytes:
        """
        Pillow를 활용하여 다양한 색상과 도형의 합성 이미지를 동적으로 생성합니다.
        (네트워크 통신 없이 100% 로컬에서 동작하는 벤치마크 데이터셋 구축용)
        """
        img = Image.new("RGB", (224, 224), color="white")
        draw = ImageDraw.Draw(img)
        
        # 색상 매핑
        colors = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "pink": (255, 192, 203),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
            "cyan": (0, 255, 255),
            "brown": (139, 69, 19),
            "black": (0, 0, 0)
        }
        fill_color = colors.get(color, (128, 128, 128))

        if shape == "circle":
            draw.ellipse([40, 40, 184, 184], fill=fill_color)
        elif shape == "square":
            draw.rectangle([40, 40, 184, 184], fill=fill_color)
        elif shape == "triangle":
            draw.polygon([(112, 40), (40, 184), (184, 184)], fill=fill_color)
        elif shape == "ring":
            draw.ellipse([40, 40, 184, 184], outline=fill_color, width=20)
        elif shape == "cross":
            draw.rectangle([92, 40, 132, 184], fill=fill_color)
            draw.rectangle([40, 92, 184, 132], fill=fill_color)
        elif shape == "star":
            draw.polygon([(112, 30), (130, 90), (194, 112), (130, 134), (112, 194), (94, 134), (30, 112), (94, 90)], fill=fill_color)
        else: # dot
            draw.ellipse([92, 92, 132, 132], fill=fill_color)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def calculate_metrics(self, sim_matrix: np.ndarray) -> Dict[str, float]:
        """
        유사도 행렬로부터 Recall 및 NDCG 메트릭을 계산합니다.
        행 = 쿼리, 열 = 대상 포스트 (대각선 원소가 정답 매칭)
        """
        n = sim_matrix.shape[0]
        k_list = [1, 3, 5]
        results = {f"Recall@{k}": [] for k in k_list}
        for k in k_list:
            results[f"NDCG@{k}"] = []

        for i in range(n):
            scores = sim_matrix[i]
            ranks = np.argsort(scores)[::-1]
            
            for k in k_list:
                top_k = ranks[:k]
                hit = 1.0 if i in top_k else 0.0
                results[f"Recall@{k}"].append(hit)

                if i in top_k:
                    rank_idx = np.where(top_k == i)[0][0] + 1
                    results[f"NDCG@{k}"].append(1.0 / np.log2(rank_idx + 1))
                else:
                    results[f"NDCG@{k}"].append(0.0)

        return {metric: float(np.mean(vals)) for metric, vals in results.items()}

    async def run_synthetic_benchmark(self) -> Dict[str, Any]:
        """
        10가지 형태 및 색상의 합성 데이터셋을 실시간 생성하여 CLIP/SBERT 모델 성능을 측정합니다.
        """
        dataset_defs = [
            ("circle", "red", "a red circle drawn on a white background"),
            ("square", "blue", "a blue square shape on white backdrop"),
            ("triangle", "green", "a green triangle shape on white background"),
            ("ring", "yellow", "a yellow ring outline on white background"),
            ("cross", "pink", "a pink cross shape drawn on white background"),
            ("star", "purple", "a purple eight pointed star on white background"),
            ("circle", "orange", "an orange circle shape on white background"),
            ("square", "cyan", "a cyan color square drawn on white background"),
            ("triangle", "brown", "a brown triangle shape on white background"),
            ("cross", "black", "a black cross shape on white background")
        ]

        images = []
        captions = []
        
        for shape, color, caption in dataset_defs:
            images.append(self.create_synthetic_image(shape, color))
            captions.append(caption)

        img_embs = []
        for img_bytes in images:
            emb = nlp_embedder.embed_image(img_bytes).to(self.device)
            img_embs.append(F.normalize(emb, p=2, dim=-1).cpu().numpy())

        img_embs = np.array(img_embs) # [10, 512]

        # CLIP 텍스트 임베딩 계산
        clip_txt_embs = []
        for text in captions:
            inputs = nlp_embedder.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = nlp_embedder.clip_model.get_text_features(**inputs)
            clip_txt_embs.append(F.normalize(text_features, p=2, dim=-1).squeeze(0).cpu().numpy())

        clip_txt_embs = np.array(clip_txt_embs) # [10, 512]

        # Text-to-Image 유사도
        t2i_sim = np.dot(clip_txt_embs, img_embs.T)
        t2i_metrics = self.calculate_metrics(t2i_sim)

        # Image-to-Image 유사도 (약간 변형된 이미지 쿼리 사용)
        query_images = []
        for shape, color, _ in dataset_defs:
            img = Image.new("RGB", (224, 224), color="white")
            draw = ImageDraw.Draw(img)
            colors = {
                "red": (240, 20, 20),
                "blue": (20, 20, 240),
                "green": (20, 240, 20),
                "yellow": (245, 245, 10),
                "pink": (250, 180, 190),
                "purple": (120, 10, 120),
                "orange": (245, 150, 10),
                "cyan": (10, 245, 245),
                "brown": (130, 60, 10),
                "black": (10, 10, 10)
            }
            fill_color = colors.get(color, (120, 120, 120))
            if shape == "circle":
                draw.ellipse([45, 45, 179, 179], fill=fill_color)
            elif shape == "square":
                draw.rectangle([45, 45, 179, 179], fill=fill_color)
            elif shape == "triangle":
                draw.polygon([(112, 45), (45, 179), (179, 179)], fill=fill_color)
            elif shape == "ring":
                draw.ellipse([45, 45, 179, 179], outline=fill_color, width=15)
            elif shape == "cross":
                draw.rectangle([95, 45, 129, 179], fill=fill_color)
                draw.rectangle([45, 95, 179, 129], fill=fill_color)
            elif shape == "star":
                draw.polygon([(112, 35), (128, 85), (189, 107), (128, 129), (112, 189), (96, 129), (35, 107), (96, 85)], fill=fill_color)
            else:
                draw.ellipse([95, 95, 129, 129], fill=fill_color)
            
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            query_images.append(buf.getvalue())

        q_img_embs = []
        for q_bytes in query_images:
            emb = nlp_embedder.embed_image(q_bytes).to(self.device)
            q_img_embs.append(F.normalize(emb, p=2, dim=-1).cpu().numpy())

        q_img_embs = np.array(q_img_embs) # [10, 512]

        i2i_sim = np.dot(q_img_embs, img_embs.T)
        i2i_metrics = self.calculate_metrics(i2i_sim)

        return {
            "status": "success",
            "dataset_name": "Synthetic Shape-Color Dataset (10 Classes)",
            "sample_size": 10,
            "text_to_image": t2i_metrics,
            "image_to_image": i2i_metrics
        }

    async def run_custom_benchmark(self, zip_bytes: bytes) -> Dict[str, Any]:
        """
        사용자가 직접 업로드한 ZIP 파일(images + dataset.json)을 활용해 정확도를 측정합니다.
        """
        try:
            images_data = {}
            metadata = []

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                json_file = None
                for name in z.namelist():
                    if name.endswith("dataset.json"):
                        json_file = name
                        break
                
                if not json_file:
                    raise ValueError("ZIP archive must contain 'dataset.json'")
                
                metadata = json.loads(z.read(json_file).decode("utf-8"))
                
                for item in metadata:
                    filename = item["filename"]
                    zip_path = None
                    for name in z.namelist():
                        if name.endswith(filename):
                            zip_path = name
                            break
                    if zip_path:
                        images_data[filename] = z.read(zip_path)

            if not metadata or not images_data:
                raise ValueError("No valid image metadata or images found in ZIP")

            img_embs = []
            clip_txt_embs = []
            captions_list = []
            filenames_list = []

            for item in metadata:
                filename = item["filename"]
                caption = item["caption"]
                if filename not in images_data:
                    continue

                img_bytes = images_data[filename]
                
                img_emb = nlp_embedder.embed_image(img_bytes).to(self.device)
                img_embs.append(F.normalize(img_emb, p=2, dim=-1).cpu().numpy())
                
                inputs = nlp_embedder.processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    text_features = nlp_embedder.clip_model.get_text_features(**inputs)
                clip_txt_embs.append(F.normalize(text_features, p=2, dim=-1).squeeze(0).cpu().numpy())

                captions_list.append(caption)
                filenames_list.append(filename)

            n_samples = len(img_embs)
            if n_samples < 2:
                raise ValueError("Benchmark requires at least 2 image samples to measure recall")

            img_embs = np.array(img_embs)
            clip_txt_embs = np.array(clip_txt_embs)

            t2i_sim = np.dot(clip_txt_embs, img_embs.T)
            t2i_metrics = self.calculate_metrics(t2i_sim)

            i2i_self_sim = np.dot(img_embs, img_embs.T)
            np.fill_diagonal(i2i_self_sim, 0.0)
            avg_cross_sim = float(np.sum(i2i_self_sim) / (n_samples * (n_samples - 1)))

            return {
                "status": "success",
                "dataset_name": "Custom Uploaded Dataset",
                "sample_size": n_samples,
                "text_to_image": t2i_metrics,
                "average_cross_image_similarity": avg_cross_sim,
                "details": [
                    {"filename": filenames_list[i], "caption": captions_list[i]}
                    for i in range(min(n_samples, 5))
                ]
            }

        except Exception as e:
            logger.error(f"❌ Custom benchmark error: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    async def inject_animal_dataset(self, db) -> int:
        """
        동물 이미지 데이터셋을 DB(upload 및 search 스키마)와 Redis에 주입합니다.
        """
        check_stmt = text(
            "SELECT COUNT(*) FROM search.post_vectors "
            "WHERE (content_text->>'is_animal_benchmark')::boolean = true"
        )
        res = await db.execute(check_stmt)
        count = res.scalar() or 0
        if count >= 10:
            logger.info("🐾 Animal dataset is already fully injected. Skipping injection.")
            return 0

        system_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        
        try:
            check_user = await db.execute(text("SELECT id FROM auth.users WHERE id = :uid"), {"uid": system_user_id})
            if not check_user.first():
                await db.execute(text(
                    "INSERT INTO auth.users (id, email, password_hash, nickname, role, is_active, created_at) "
                    "VALUES (:uid, 'system_benchmark@insterest.ai', 'N/A', 'AnimalBenchmarkSystem', 'user', true, NOW())"
                ), {"uid": system_user_id})
                await db.commit()
        except Exception as ue:
            logger.warning(f"Failed to check/create system user: {ue}")

        injected = 0
        for item in ANIMAL_DATASET:
            post_id = uuid.uuid4()
            content_id = uuid.uuid4()
            media_id = uuid.uuid4()
            
            try:
                await db.execute(text(
                    "INSERT INTO upload.content (id, user_id, content_type, metadata, created_at, is_deleted, is_ai) "
                    "VALUES (:id, :user_id, 'PHOTO', :metadata, NOW(), false, false)"
                ), {
                    "id": content_id,
                    "user_id": system_user_id,
                    "metadata": json.dumps({"is_animal_benchmark": True, "animal_name": item["name"]})
                })
                
                await db.execute(text(
                    "INSERT INTO upload.post (id, content_id, user_id, caption, like_count, view_count, created_at, is_deleted) "
                    "VALUES (:id, :content_id, :user_id, :caption, 0, 0, NOW(), false)"
                ), {
                    "id": post_id,
                    "content_id": content_id,
                    "user_id": system_user_id,
                    "caption": item["caption"]
                })
                
                await db.execute(text(
                    "INSERT INTO upload.media (id, user_id, content_id, type, url, created_at, metadata_info) "
                    "VALUES (:id, :user_id, :content_id, 'IMAGE', :url, NOW(), '{}'::jsonb)"
                ), {
                    "id": media_id,
                    "user_id": system_user_id,
                    "content_id": content_id,
                    "url": item["url"]
                })
                
                await db.commit()
            except Exception as dbe:
                logger.error(f"❌ Failed to insert upload DB records for animal {item['name']}: {dbe}")
                await db.rollback()
                continue

            img_bytes = None
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    img_res = await client.get(item["url"])
                    if img_res.status_code == 200:
                        img_bytes = img_res.content
            except Exception as ne:
                logger.warning(f"Internet download failed for {item['name']}, using synthetic fallback: {ne}")

            if not img_bytes:
                try:
                    img = Image.new("RGB", (224, 224), color="lightblue")
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([20, 20, 204, 204], outline="darkblue", width=4)
                    draw.text((40, 100), item["name"].upper(), fill="darkblue")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()
                except Exception as pe:
                    logger.error(f"Pillow drawing failed: {pe}")
                    img_bytes = b""

            caption_emb = nlp_embedder.embed_text(item["caption"]).to(self.device)
            caption_vec = F.normalize(caption_emb, p=2, dim=-1).cpu().numpy()
            
            if item["hashtags"]:
                h_embs = nlp_embedder.embed_batch(item["hashtags"])
                h_emb = torch.mean(h_embs, dim=0).to(self.device)
                hashtag_vec = F.normalize(h_emb, p=2, dim=-1).cpu().numpy()
            else:
                hashtag_vec = np.zeros(768, dtype=np.float32)

            if img_bytes:
                img_emb = nlp_embedder.embed_image(img_bytes).to(self.device)
                image_vec = F.normalize(img_emb, p=2, dim=-1).cpu().numpy()
            else:
                image_vec = np.zeros(512, dtype=np.float32)

            from app.services.intelligence import intel_service
            try:
                await intel_service.index_post(
                    db=db,
                    post_id=post_id,
                    caption_vec=caption_vec,
                    hashtag_vec=hashtag_vec,
                    image_vec=image_vec,
                    metadata={"caption": item["caption"], "is_animal_benchmark": True, "animal_name": item["name"]}
                )
                injected += 1
            except Exception as ie:
                logger.error(f"❌ Indexing failed for animal {item['name']}: {ie}")

        return injected

    async def run_animal_db_benchmark(self, db) -> Dict[str, Any]:
        """
        동물분류 데이터셋을 DB에 주입하고 다대다 교차 매칭 벤치마크를 수행합니다.
        """
        injected_count = await self.inject_animal_dataset(db)
        logger.info(f"🐾 Animal dataset verification done. Injected {injected_count} new posts.")

        from app.services.intelligence import intel_service
        stmt = text(
            "SELECT post_id, content_text FROM search.post_vectors "
            "WHERE (content_text->>'is_animal_benchmark')::boolean = true"
        )
        res = await db.execute(stmt)
        rows = res.all()
        
        db_animals = {}
        for r in rows:
            post_id = r[0]
            meta = r[1]
            animal_name = meta.get("animal_name") if isinstance(meta, dict) else None
            if animal_name:
                db_animals[animal_name] = post_id

        if len(db_animals) < 2:
            return {
                "status": "error",
                "message": f"Insufficient benchmark samples in DB. Found {len(db_animals)}, required at least 2."
            }

        details = []
        text_to_image_hits = {1: [], 3: [], 5: []}
        image_to_image_hits = {1: [], 3: [], 5: []}
        
        for item in ANIMAL_DATASET:
            name = item["name"]
            target_post_id = db_animals.get(name)
            if not target_post_id:
                continue

            txt_results = await intel_service.discover(
                db=db,
                query_text=item["caption"],
                limit=10,
                use_personalization=False
            )
            txt_ids = [uuid.UUID(r["id"]) if isinstance(r["id"], str) else r["id"] for r in txt_results]
            
            text_rank = 999
            if target_post_id in txt_ids:
                text_rank = txt_ids.index(target_post_id) + 1
            
            for k in [1, 3, 5]:
                text_to_image_hits[k].append(1.0 if text_rank <= k else 0.0)

            img_bytes = None
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    img_res = await client.get(item["url"])
                    if img_res.status_code == 200:
                        img_bytes = img_res.content
            except Exception:
                pass

            if not img_bytes:
                try:
                    img = Image.new("RGB", (224, 224), color="lightblue")
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([20, 20, 204, 204], outline="darkblue", width=4)
                    draw.text((40, 100), item["name"].upper(), fill="darkblue")
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    img_bytes = buf.getvalue()
                except Exception:
                    img_bytes = b""

            img_results = await intel_service.discover_by_image(
                db=db,
                image_bytes=img_bytes,
                limit=10,
                use_personalization=False
            )
            img_ids = [uuid.UUID(r["id"]) if isinstance(r["id"], str) else r["id"] for r in img_results]

            image_rank = 999
            if target_post_id in img_ids:
                image_rank = img_ids.index(target_post_id) + 1

            for k in [1, 3, 5]:
                image_to_image_hits[k].append(1.0 if image_rank <= k else 0.0)

            details.append({
                "name": name,
                "caption": item["caption"],
                "url": item["url"],
                "text_rank": text_rank if text_rank < 999 else "Not Found (10+)",
                "image_rank": image_rank if image_rank < 999 else "Not Found (10+)"
            })

        t2i_metrics = {
            f"Recall@{k}": float(np.mean(text_to_image_hits[k])) for k in [1, 3, 5]
        }
        i2i_metrics = {
            f"Recall@{k}": float(np.mean(image_to_image_hits[k])) for k in [1, 3, 5]
        }

        ndcg_map = {1: 1.0, 2: 0.63, 3: 0.5, 4: 0.43, 5: 0.38}
        t2i_metrics["NDCG@3"] = float(np.mean([ndcg_map.get(det["text_rank"], 0.0) if isinstance(det["text_rank"], int) and det["text_rank"] <= 3 else 0.0 for det in details]))
        i2i_metrics["NDCG@3"] = float(np.mean([ndcg_map.get(det["image_rank"], 0.0) if isinstance(det["image_rank"], int) and det["image_rank"] <= 3 else 0.0 for det in details]))

        return {
            "status": "success",
            "dataset_name": "Famous Animal Classification Dataset (10 Classes)",
            "sample_size": len(details),
            "text_to_image": t2i_metrics,
            "image_to_image": i2i_metrics,
            "details": details
        }

ANIMAL_DATASET = [
    {
        "name": "dog",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/18/Dog_Chihuahua.jpg",
        "caption": "a cute small brown chihuahua dog sitting on the floor looking at the camera",
        "hashtags": ["dog", "puppy", "chihuahua", "animal"]
    },
    {
        "name": "cat",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "caption": "a close up portrait of a cute domestic tabby cat with bright green eyes",
        "hashtags": ["cat", "kitty", "pet", "animal"]
    },
    {
        "name": "tiger",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/56/Bengal_Tiger_in_a_national_park.jpg",
        "caption": "a majestic wild bengal tiger with orange and black stripes walking in the green forest",
        "hashtags": ["tiger", "wildcat", "predator", "animal"]
    },
    {
        "name": "lion",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/73/Lion_waiting_in_Namibia.jpg",
        "caption": "a powerful male lion with a large dark mane sitting in the dry yellow savanna grass",
        "hashtags": ["lion", "king", "savanna", "animal"]
    },
    {
        "name": "elephant",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg",
        "caption": "a massive grey african bush elephant standing in the grass field under sunlight",
        "hashtags": ["elephant", "mammal", "safari", "animal"]
    },
    {
        "name": "panda",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/0f/Grosser_Panda.JPG",
        "caption": "a cute giant panda bear sitting down and eating green bamboo leaves",
        "hashtags": ["panda", "bear", "china", "animal"]
    },
    {
        "name": "giraffe",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Giraffe_standing.jpg",
        "caption": "a very tall giraffe with brown spot patterns standing near green acacia trees",
        "hashtags": ["giraffe", "tall", "safari", "animal"]
    },
    {
        "name": "zebra",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/07/Plains_Zebra_Equus_quagga_at_Etosha_National_Park.jpg",
        "caption": "a wild plains zebra with distinct black and white stripes standing in dry grass",
        "hashtags": ["zebra", "stripes", "safari", "animal"]
    },
    {
        "name": "bear",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/71/2010-kodiak-bear-1.jpg",
        "caption": "a large brown kodiak bear standing in wild green forest stream searching for fish",
        "hashtags": ["bear", "grizzly", "wildlife", "animal"]
    },
    {
        "name": "monkey",
        "url": "https://upload.wikimedia.org/wikipedia/commons/4/43/Bonobo_sitting.jpg",
        "caption": "a black bonobo monkey sitting on the ground looking forward with human like eyes",
        "hashtags": ["monkey", "ape", "bonobo", "animal"]
    }
]

benchmark_service = BenchmarkService()
