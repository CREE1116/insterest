import io
import asyncio
import json
import os
import random
import zipfile
import uuid
import numpy as np
import torch
import torch.nn.functional as F
import httpx
from PIL import Image, ImageDraw
from typing import List, Dict, Any, Tuple
from sqlalchemy import text
import logging
from app.ml.nlp import nlp_embedder

logger = logging.getLogger(__name__)

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

        # CLIP 텍스트 임베딩 계산 (SentenceTransformer API)
        with torch.no_grad():
            clip_txt_tensor = nlp_embedder.clip_model.encode(captions, convert_to_tensor=True).to(self.device)
        clip_txt_embs = F.normalize(clip_txt_tensor, p=2, dim=-1).cpu().numpy()  # [10, 512]

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
                
                with torch.no_grad():
                    text_features = nlp_embedder.clip_model.encode(caption, convert_to_tensor=True).to(self.device)
                clip_txt_embs.append(F.normalize(text_features.unsqueeze(0), p=2, dim=-1).squeeze(0).cpu().numpy())

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

    # ── CIFAR-10 Benchmark ────────────────────────────────────────────────────

    CIFAR100_SAMPLES_PER_CLASS = 2  # 100 classes × 2 = 200 total samples

    def _cifar_img_to_bytes(self, pil_img) -> bytes:
        img = pil_img.resize((224, 224), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    def _load_cifar100_samples(self) -> Dict[str, list]:
        """CIFAR-100 다운로드 후 클래스별 PIL 이미지 N장 반환."""
        import torchvision.datasets as tvd
        dataset = tvd.CIFAR100(root="/tmp/cifar100", train=True, download=True)
        class_names = dataset.classes  # 100 fine-grained classes
        per_class: Dict[str, list] = {c: [] for c in class_names}
        for img, label in dataset:
            name = class_names[label]
            if len(per_class[name]) < self.CIFAR100_SAMPLES_PER_CLASS:
                per_class[name].append(img)
            if all(len(v) >= self.CIFAR100_SAMPLES_PER_CLASS for v in per_class.values()):
                break
        return per_class

    async def inject_cifar_dataset(self, db, per_class=None) -> Dict[str, Dict[int, uuid.UUID]]:
        """CIFAR-100을 DB와 Redis에 주입. 이미 존재하는 데이터는 재사용, 없는 것만 삽입."""
        system_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        try:
            check_user = await db.execute(text("SELECT id FROM auth.users WHERE id = :uid"), {"uid": system_user_id})
            if not check_user.first():
                await db.execute(text(
                    "INSERT INTO auth.users (id, email, password_hash, nickname, role, is_active, created_at) "
                    "VALUES (:uid, 'system_benchmark@insterest.ai', 'N/A', 'BenchmarkSystem', 'user', true, NOW())"
                ), {"uid": system_user_id})
                await db.commit()
        except Exception as e:
            logger.warning(f"System user upsert: {e}")

        # Load existing benchmark posts from DB.
        # Query upload.content directly (no JOIN on post_vectors) so posts whose
        # vector indexing previously failed are still detected as existing and not
        # re-inserted as duplicates.
        existing: Dict[str, Dict[int, uuid.UUID]] = {}
        try:
            res = await db.execute(text("""
                SELECT p.id, c.metadata::jsonb
                FROM upload.post p
                JOIN upload.content c ON c.id = p.content_id
                WHERE p.user_id = :uid
                  AND p.is_deleted = FALSE
                  AND (c.metadata::jsonb->>'is_animal_benchmark')::boolean = true
            """), {"uid": system_user_id})
            for row in res.all():
                pid, meta = row
                if isinstance(meta, dict):
                    cls = meta.get("animal_name")
                    idx = meta.get("sample_idx")
                    if cls is not None and idx is not None:
                        existing.setdefault(cls, {})[int(idx)] = pid
        except Exception as e:
            logger.warning(f"Could not load existing benchmark data: {e}")

        n_existing = sum(len(v) for v in existing.values())
        if n_existing:
            logger.info(f"♻️ Found {n_existing} existing benchmark posts ({len(existing)} classes). Injecting missing only...")

        if per_class is None:
            logger.info("📥 Loading CIFAR-100 dataset...")
            per_class = await asyncio.to_thread(self._load_cifar100_samples)

        from app.services.intelligence import intel_service
        class_post_ids: Dict[str, Dict[int, uuid.UUID]] = {k: dict(v) for k, v in existing.items()}
        total_new = 0

        # Directory where CIFAR images are saved for static serving
        img_dir = "/data/cifar_images"
        os.makedirs(img_dir, exist_ok=True)

        for class_name, imgs in per_class.items():
            if class_name not in class_post_ids:
                class_post_ids[class_name] = {}

            # 클래스당 해시태그 1회 upsert
            try:
                await db.execute(text(
                    "INSERT INTO upload.hashtag (tag) VALUES (:tag) ON CONFLICT (tag) DO NOTHING"
                ), {"tag": class_name})
                await db.commit()
                h_res = await db.execute(text(
                    "SELECT id FROM upload.hashtag WHERE tag = :tag"
                ), {"tag": class_name})
                hashtag_id = h_res.scalar()
            except Exception as e:
                logger.warning(f"Hashtag upsert failed for {class_name}: {e}")
                await db.rollback()
                hashtag_id = None

            for idx, pil_img in enumerate(imgs):
                img_bytes = await asyncio.to_thread(self._cifar_img_to_bytes, pil_img)
                caption = f"{class_name}, sample {idx + 1}"
                hashtags = [class_name]

                # Save image file for static serving (idempotent)
                img_file = os.path.join(img_dir, f"{class_name}_{idx}.jpg")
                if not os.path.exists(img_file):
                    try:
                        with open(img_file, "wb") as f:
                            f.write(img_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to save benchmark image {img_file}: {e}")

                if idx in class_post_ids[class_name]:
                    # DB exists — verify Redis has image_vector; fix if missing
                    existing_post_id = class_post_ids[class_name][idx]
                    from app.ml.vector_store import vector_store as _vs
                    if not _vs.r.hexists(f"post:{existing_post_id}", "image_vector"):
                        logger.info(f"🔄 Re-indexing {class_name}[{idx}] (image_vector missing from Redis)")
                        i_vec = F.normalize(nlp_embedder.embed_image(img_bytes).to(self.device), p=2, dim=-1).cpu().numpy()
                        _vs.r.hset(f"post:{existing_post_id}", mapping={"image_vector": i_vec.astype(np.float32).tobytes()})
                    continue

                # Race-condition guard: re-check DB right before insert
                dup_check = await db.execute(text("""
                    SELECT p.id FROM upload.post p
                    JOIN upload.content c ON c.id = p.content_id
                    WHERE p.user_id = :uid AND p.is_deleted = FALSE
                      AND (c.metadata::jsonb->>'animal_name') = :cls
                      AND (c.metadata::jsonb->>'sample_idx')::int = :sidx
                    LIMIT 1
                """), {"uid": system_user_id, "cls": class_name, "sidx": idx})
                existing_row = dup_check.first()
                if existing_row:
                    class_post_ids[class_name][idx] = existing_row[0]
                    continue

                post_id = uuid.uuid4()
                content_id = uuid.uuid4()
                media_id = uuid.uuid4()

                try:
                    await db.execute(text(
                        "INSERT INTO upload.content (id, user_id, content_type, metadata, created_at, is_deleted, is_ai) "
                        "VALUES (:id, :uid, 'PHOTO', :meta, NOW(), false, false)"
                    ), {"id": content_id, "uid": system_user_id,
                        "meta": json.dumps({"is_animal_benchmark": True, "animal_name": class_name, "sample_idx": idx})})
                    await db.execute(text(
                        "INSERT INTO upload.post (id, content_id, user_id, caption, like_count, view_count, created_at, is_deleted) "
                        "VALUES (:id, :cid, :uid, :cap, 0, 0, NOW(), false)"
                    ), {"id": post_id, "cid": content_id, "uid": system_user_id, "cap": caption})
                    await db.execute(text(
                        "INSERT INTO upload.media (id, user_id, content_id, type, url, created_at, metadata_info) "
                        "VALUES (:id, :uid, :cid, 'IMAGE', :url, NOW(), '{}'::jsonb)"
                    ), {"id": media_id, "uid": system_user_id, "cid": content_id,
                        "url": f"/benchmark/cifar100/{class_name}_{idx}.jpg"})
                    # 해시태그 연결
                    if hashtag_id is not None:
                        await db.execute(text(
                            "INSERT INTO upload.post_hashtag (post_id, hashtag_id) "
                            "VALUES (:pid, :hid) ON CONFLICT DO NOTHING"
                        ), {"pid": post_id, "hid": hashtag_id})
                    await db.commit()
                except Exception as e:
                    logger.error(f"DB insert failed for {class_name}[{idx}]: {e}")
                    await db.rollback()
                    continue

                c_vec = F.normalize(nlp_embedder.embed_text_clip(caption).to(self.device), p=2, dim=-1).cpu().numpy()
                h_embs = nlp_embedder.embed_batch_clip(hashtags)
                h_vec = F.normalize(torch.mean(h_embs, dim=0).to(self.device), p=2, dim=-1).cpu().numpy()
                i_vec = F.normalize(nlp_embedder.embed_image(img_bytes).to(self.device), p=2, dim=-1).cpu().numpy()
                try:
                    await intel_service.index_post(
                        db=db, post_id=post_id,
                        caption_vec=c_vec, hashtag_vec=h_vec, image_vec=i_vec,
                        metadata={"caption": caption, "is_animal_benchmark": True,
                                  "animal_name": class_name, "sample_idx": idx}
                    )
                    class_post_ids[class_name][idx] = post_id
                    total_new += 1
                except Exception as e:
                    logger.error(f"Redis index failed for {class_name}[{idx}]: {e}")

        logger.info(f"✅ Benchmark dataset ready. {n_existing} reused + {total_new} newly injected.")

        # Seed interactions for all benchmark posts (idempotent)
        await self._seed_benchmark_interactions(db, class_post_ids)

        return class_post_ids

    async def _seed_benchmark_interactions(self, db, class_post_ids: Dict[str, Dict[int, uuid.UUID]]) -> None:
        """동물 벤치마크 포스트에 가상 좋아요 시드. 이미 있으면 스킵 (idempotent)."""
        all_pids = [pid for pids in class_post_ids.values() for pid in pids.values()]
        if not all_pids:
            return

        # Idempotency check: if any of the benchmark posts already have likes, skip
        try:
            res = await db.execute(
                text("SELECT COUNT(*) FROM interaction.likes WHERE post_id = :pid"),
                {"pid": all_pids[0]}
            )
            if (res.scalar() or 0) > 0:
                logger.info("⏭️ Benchmark interactions already seeded. Skipping.")
                return
        except Exception as e:
            logger.warning(f"Interaction check failed: {e}")

        # 20 synthetic animal-lover personas
        persona_names = [
            "PetPhotographer", "WildlifeExplorer", "ZooKeeper", "AnimalRescuer",
            "NatureLover", "CutePetAddict", "ExoticAnimalFan", "FarmLifeLover",
            "MarineLifeEnthusiast", "BirdWatcher", "CatObsessed", "DogMom",
            "RabbitsForever", "FoxyFollower", "TigerLover", "BearGrills",
            "AquaticAdmirer", "HorsePerson", "MicroFaunaFan", "SafariSeeker",
        ]

        seeded = 0
        for name in persona_names:
            user_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"benchmark_{name}")
            liked_pids = random.sample(all_pids, max(1, len(all_pids) * 4 // 10))
            for i, pid in enumerate(liked_pids):
                try:
                    await db.execute(text("""
                        INSERT INTO interaction.likes (user_id, post_id, created_at)
                        VALUES (:u, :p, NOW() - INTERVAL '1 second' * :offset)
                        ON CONFLICT DO NOTHING
                    """), {"u": user_id, "p": pid, "offset": i})
                    seeded += 1
                except Exception as e:
                    logger.warning(f"Interaction seed failed: {e}")

        await db.commit()
        logger.info(f"🎭 Seeded {seeded} benchmark interactions across {len(persona_names)} personas.")

    async def run_animal_db_benchmark(self, db) -> Dict[str, Any]:
        """CIFAR-100 실제 이미지로 크로스모달 검색 정확도 벤치마크. 데이터는 유지된다."""
        logger.info("📥 Loading CIFAR-100 dataset...")
        per_class = await asyncio.to_thread(self._load_cifar100_samples)
        class_post_ids = await self.inject_cifar_dataset(db, per_class=per_class)
        all_post_ids = {pid for pids in class_post_ids.values() for pid in pids.values()}
        if len(all_post_ids) < 10:
            return {"status": "error", "message": f"Insufficient samples injected ({len(all_post_ids)})."}

        from app.services.intelligence import intel_service

        # Train UserTower on injected benchmark data before measuring
        try:
            logger.info("🏋️ Training UserTower on injected benchmark data...")
            await intel_service.train_discovery(db)
            logger.info("✅ Training complete.")
        except Exception as e:
            logger.warning(f"⚠️ Training failed, continuing with current model: {e}")

        total_samples = len(all_post_ids)
        search_limit = min(total_samples, 200)

        text_hits = {1: [], 3: [], 5: []}
        image_hits = {1: [], 3: [], 5: []}
        details = []

        for class_name, idx_to_pid in class_post_ids.items():
            class_pids = set(idx_to_pid.values())
            n = len(class_pids)

            # Text→Image: query = class label, check how many class images rank in top-k
            txt_results = await intel_service.discover(
                db=db, query_text=class_name, limit=search_limit, use_personalization=False
            )
            txt_ids = [uuid.UUID(r["id"]) if isinstance(r["id"], str) else r["id"] for r in txt_results]
            txt_ranks = sorted([txt_ids.index(p) + 1 for p in class_pids if p in txt_ids])
            best_text_rank = txt_ranks[0] if txt_ranks else 999
            for k in [1, 3, 5]:
                text_hits[k].append(sum(1 for r in txt_ranks if r <= k) / n)

            # Image→Image: first sample → find other samples of same class
            best_image_rank = 999
            if 0 in idx_to_pid and per_class.get(class_name):
                img_bytes = await asyncio.to_thread(self._cifar_img_to_bytes, per_class[class_name][0])
                img_results = await intel_service.discover_by_image(
                    db=db, image_bytes=img_bytes, limit=search_limit, use_personalization=False
                )
                img_ids = [uuid.UUID(r["id"]) if isinstance(r["id"], str) else r["id"] for r in img_results]
                other_pids = class_pids - {idx_to_pid[0]}
                img_ranks = sorted([img_ids.index(p) + 1 for p in other_pids if p in img_ids])
                best_image_rank = img_ranks[0] if img_ranks else 999
                for k in [1, 3, 5]:
                    image_hits[k].append(sum(1 for r in img_ranks if r <= k) / max(len(other_pids), 1))
            else:
                for k in [1, 3, 5]:
                    image_hits[k].append(0.0)

            details.append({
                "name": class_name,
                "caption": f"{class_name}, sample 1",
                "url": f"/benchmark/cifar100/{class_name}_0.jpg",
                "text_rank": best_text_rank if best_text_rank < 999 else "Not Found",
                "image_rank": best_image_rank if best_image_rank < 999 else "Not Found",
                "sample_count": n,
            })

        ndcg_map = {1: 1.0, 2: 0.63, 3: 0.5, 4: 0.43, 5: 0.38}
        t2i = {f"Recall@{k}": float(np.mean(text_hits[k])) for k in [1, 3, 5]}
        i2i = {f"Recall@{k}": float(np.mean(image_hits[k])) for k in [1, 3, 5]}
        t2i["NDCG@3"] = float(np.mean([
            ndcg_map.get(d["text_rank"], 0.0) if isinstance(d["text_rank"], int) and d["text_rank"] <= 3 else 0.0
            for d in details]))
        i2i["NDCG@3"] = float(np.mean([
            ndcg_map.get(d["image_rank"], 0.0) if isinstance(d["image_rank"], int) and d["image_rank"] <= 3 else 0.0
            for d in details]))

        return {
            "status": "success",
            "dataset_name": f"CIFAR-100 ({len(class_post_ids)} classes × {self.CIFAR100_SAMPLES_PER_CLASS} samples = {total_samples} items)",
            "sample_size": total_samples,
            "text_to_image": t2i,
            "image_to_image": i2i,
            "details": details,
        }

    async def _cleanup_benchmark(self, db, post_ids: set) -> None:
        """벤치마크 완료 후 DB와 Redis에서 주입된 데이터 전체 삭제."""
        system_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        try:
            from app.services.intelligence import intel_service
            for pid in post_ids:
                try:
                    await intel_service.remove_post(pid)
                except Exception as e:
                    logger.warning(f"Redis remove failed for {pid}: {e}")

            await db.execute(text("DELETE FROM upload.media WHERE user_id = :uid"), {"uid": system_user_id})
            await db.execute(text("DELETE FROM upload.post WHERE user_id = :uid"), {"uid": system_user_id})
            await db.execute(text("DELETE FROM upload.content WHERE user_id = :uid"), {"uid": system_user_id})
            await db.execute(text(
                "DELETE FROM search.post_vectors WHERE (content_text->>'is_animal_benchmark')::boolean = true"
            ))
            await db.commit()
            logger.info(f"🧹 Cleaned up {len(post_ids)} benchmark posts from DB and Redis.")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            await db.rollback()

    async def delete_benchmark_data(self, db) -> Dict[str, Any]:
        """DevConsole에서 호출: 모든 벤치마크 데이터(DB + Redis + 좋아요) 완전 삭제."""
        system_user_id = uuid.UUID("00000000-0000-0000-0000-000000000000")
        try:
            from app.services.intelligence import intel_service
            from app.ml.vector_store import vector_store

            # Gather post IDs before deletion so we can clean Redis
            res = await db.execute(text("SELECT id FROM upload.post WHERE user_id = :uid"), {"uid": system_user_id})
            post_ids = [row[0] for row in res.all()]

            for pid in post_ids:
                try:
                    await intel_service.remove_post(pid)
                except Exception as e:
                    logger.warning(f"Redis remove failed for {pid}: {e}")

            # Delete in FK-safe order
            await db.execute(text(
                "DELETE FROM interaction.likes WHERE post_id IN (SELECT id FROM upload.post WHERE user_id = :uid)"
            ), {"uid": system_user_id})
            await db.execute(text(
                "DELETE FROM search.post_vectors WHERE (content_text->>'is_animal_benchmark')::boolean = true"
            ))
            await db.execute(text("DELETE FROM upload.media WHERE user_id = :uid"), {"uid": system_user_id})
            await db.execute(text(
                "DELETE FROM upload.post_hashtag WHERE post_id IN (SELECT id FROM upload.post WHERE user_id = :uid)"
            ), {"uid": system_user_id})
            await db.execute(text("DELETE FROM upload.post WHERE user_id = :uid"), {"uid": system_user_id})
            await db.execute(text("DELETE FROM upload.content WHERE user_id = :uid"), {"uid": system_user_id})
            await db.commit()

            logger.info(f"🗑️ Deleted {len(post_ids)} benchmark posts + likes from DB and Redis.")
            return {"status": "success", "deleted_posts": len(post_ids)}
        except Exception as e:
            logger.error(f"❌ delete_benchmark_data failed: {e}", exc_info=True)
            await db.rollback()
            return {"status": "error", "message": str(e)}


benchmark_service = BenchmarkService()
