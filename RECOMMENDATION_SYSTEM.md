# Insterest Recommendation System

`recommendation_service`의 멀티모달 추천 및 탐색 시스템 아키텍처를 설명합니다.

---

## 1. 개요

사용자가 포스트를 업로드하면 CLIP과 SBERT 두 임베딩 모델이 각각 다른 역할을 맡아 벡터를 생성하고, Redis의 세 개 HNSW 인덱스에 저장합니다. 검색 요청이 들어오면 RRF(Reciprocal Rank Fusion)를 통해 텍스트 검색 랭킹과 개인화 추천 랭킹을 점수 수준에서 병합합니다.

### 모델 역할 분리

| 모델 | 차원 | 역할 |
|:---|:---:|:---|
| **CLIP ViT-B/32** | 512d | 이미지↔텍스트 통합 의미 공간. 추천·이미지 검색에 사용 |
| **SBERT all-mpnet-base-v2** | 768d | 텍스트↔텍스트 의미 유사도. 텍스트 검색에 사용 |

CLIP이 텍스트 검색에서 SBERT보다 정확도가 낮은 이유는 CLIP의 텍스트 인코더가 텍스트-이미지 정렬에 최적화되어 있기 때문입니다. 두 공간은 의미론적으로 이질적이므로 MLP 브릿지 없이 **RRF를 통해 랭킹 수준에서만 결합**합니다.

---

## 2. 벡터 저장 구조

포스트당 Redis 해시 하나(`post:{post_id}`)에 세 벡터를 함께 저장합니다.

```
Redis Hash  post:{post_id}
├── vector        [512d, FLOAT32]  — CLIP 통합 벡터 (추천/개인화)
├── text_vector   [768d, FLOAT32]  — SBERT 텍스트 벡터 (텍스트 검색)
└── image_vector  [512d, FLOAT32]  — CLIP 이미지 원본 벡터 (이미지 검색)
```

세 필드 모두 별도 HNSW 인덱스(`FT.CREATE ... DISTANCE_METRIC COSINE`)로 독립 운용됩니다. 스키마 감지 로직이 실행 시점에 기존 인덱스를 검사하고, 필드가 부족하면 자동으로 드롭 후 재생성합니다.

PostgreSQL `search.post_vectors` 테이블에는 CLIP 텍스트·해시태그·이미지 벡터를 바이너리로 보관하여 Redis 장애 시 재구축에 사용합니다.

---

## 3. 아이템 표현 (Item Tower)

학습이 필요 없는 고정 함수입니다.

```
mood_text = f"{caption} {image_prompt} {music_prompt}".strip()

CLIP_text  = CLIP.encode(mood_text)          # 512d — 캡션+이미지프롬프트+뮤직프롬프트 분위기
CLIP_image = CLIP.encode(image_bytes)        # 512d — 실제 이미지 픽셀

item_vector = normalize(CLIP_text + CLIP_image)   # 512d — Redis "vector" 필드
```

두 모달을 덧셈으로 퓨전하면 어느 한쪽이 없어도 나머지로 폴백되며, 프로젝션 레이어 없이 CLIP 원본 의미 공간을 그대로 유지합니다.

---

## 4. 유저 타워 (User Tower)

유저의 최근 좋아요 이력(최대 10개)을 512d 취향 벡터 하나로 집약합니다.

```
아키텍처:
  MultiheadAttention(embed_dim=512, num_heads=8)
  → GRU(512 → 512)
  → FC(512 → 512) + LayerNorm
  → L2 normalize

입력: [B, seq_len=10, 512]  — zero-padding으로 이력이 짧은 경우 처리
출력: [B, 512]              — 유저 취향 벡터 (cold-start시 zero vector)
```

패딩 슬롯은 어텐션 마스크로 무시되며, GRU 마지막 hidden state와 활성 슬롯의 평균 풀링을 더해 최종 벡터를 생성합니다.

---

## 5. 검색·추천 파이프라인 (RRF)

```
discover(query_text, user_id, limit)
│
├── [텍스트 검색]  SBERT.encode(query_text) → 768d
│   └── Redis KNN on text_vector  → 순위 목록 A
│
├── [개인화 추천]  UserTower(history_items) → 512d
│   └── Redis KNN on vector       → 순위 목록 B
│
└── RRF 병합
    score(d) = Σ  1 / (60 + rank_i(d))     (k=60)
    → 내림차순 정렬 → [skip : skip+limit] 슬라이스
```

- query_text 없이 user_id만 있으면 순위 목록 A가 생략되고 B만으로 추천합니다.
- user_id도 없는 cold-start 상황에서는 두 목록 모두 생략되고 트렌딩 폴백이 동작합니다.

**트렌딩 폴백 (cold-start)**

```sql
ORDER BY COALESCE(like_count, 0) * 0.7
       + COALESCE(view_count,  0) * 0.3 DESC,
         created_at DESC
LIMIT :needed
```

---

## 6. 이미지 검색 파이프라인

```
discover_by_image(image_bytes, user_id, limit)
│
├── [이미지 검색]  CLIP.encode(image_bytes) → 512d
│   └── Redis KNN on image_vector  → 순위 목록 A
│
├── [개인화 추천]  UserTower(history_items) → 512d
│   └── Redis KNN on vector        → 순위 목록 B
│
└── RRF 병합 (동일 k=60 공식)
```

---

## 7. 실시간 인덱싱 (Kafka)

포스트 생성 이벤트를 Kafka `post-created` 토픽으로 수신해 즉시 인덱싱합니다.

```
Kafka 이벤트 수신
→ mood_text 구성 (caption + image_prompt + music_prompt)
→ CLIP.encode(mood_text)     → caption_vec  512d  (DB 저장)
→ CLIP.encode(hashtags)      → hashtag_vec  512d  (DB 저장)
→ CLIP.encode(image_bytes)   → image_vec    512d  (DB 저장)
→ SBERT.encode(mood_text)    → text_vec     768d  (Redis text_vector)
→ vector_store.upsert_vector(...)            (Redis 세 필드 모두)
```

CLIP 텍스트 인코딩에는 `functools.lru_cache(maxsize=2000)`이 적용되어 반복 문자열의 인코딩을 재사용합니다.

---

## 8. 모델 학습

UserTower만 학습합니다. Item Tower는 CLIP 원본 벡터를 그대로 사용하므로 학습 불필요.

### 학습 알고리즘

**InfoNCE Loss with Hard Negative Mining**

```
입력:  user_history  [B, 10, 512]  — 유저별 이력 시퀀스
       target_item   [B, 512]      — 다음에 좋아요를 누른 아이템 벡터
       query_clip    [B, 512]      — 캡션의 CLIP 텍스트 벡터

처리:
  user_vec  = UserTower(user_history)          # [B, 512]
  query_vec = normalize(query_clip * mask)      # 50% 마스킹 — UserTower 독립성 강제
  disc_vec  = discovery(query_vec, user_vec)   # dynamic gating 퓨전

  hard_negs = 풀 전체에서 샘플링 (최대 64개, 배치 타겟 제외)
  key_vecs  = cat([target_item, hard_negs])    # [B+M, 512]

  logits    = disc_vec @ key_vecs.T / 0.07
  loss      = CrossEntropyLoss(logits, arange(B))  # 대각선이 정답
```

배치 내 동일 토픽 포스트가 서로를 밀어내는 false negative 문제를 hard negative가 완화합니다. 50% 쿼리 마스킹은 유저 타워가 쿼리 없이도 독립적으로 동작하도록 강제합니다.

**스케줄**
- 매일 00:00 자동 재학습 (24h asyncio 슬립 루프)
- 수동 트리거: `POST /api/v1/discovery/train`
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5), UserTower 파라미터만 업데이트

---

## 9. 스타트업 & 백필

```
서비스 시작 시:
  Redis count()  vs  DB COUNT(search.post_vectors)
  Redis가 부족한 경우에만 backfill_all_posts() 실행
```

백필 시 `caption_vector` 바이트 길이가 `768*4=3072`이면 구 SBERT 벡터로 감지하여 metadata에서 CLIP으로 재인코딩합니다.

---

## 10. 벤치마크

| 벤치마크 | 설명 |
|:---|:---|
| **Synthetic** | Pillow로 그린 10종 도형·색상 이미지 — CLIP text↔image Recall@1/3/5 측정 |
| **Custom ZIP** | 사용자 업로드 `dataset.json` + 이미지 — 커스텀 Recall 측정 |
| **CIFAR-100 Animal** | 100 클래스 × 2 샘플 = 200개를 DB·Redis에 주입하고 텍스트·이미지 교차 검색 정확도 평가. 이미 주입된 데이터는 재사용하고 없는 것만 삽입 (idempotent). 20개 동물 페르소나 좋아요 자동 시드. |

---

## 11. 데이터 흐름 요약

```
[포스트 생성]
  Upload Service → Kafka(post-created)
  → Kafka Consumer
  → CLIP + SBERT 인코딩
  → DB(search.post_vectors) + Redis(세 HNSW 필드) 저장

[홈 피드 요청]
  Frontend → GET /api/v1/discovery/recommend?user_id=...
  → UserTower(좋아요 이력) → CLIP 512d 취향 벡터
  → Redis KNN on "vector" → 순위 목록
  → (RRF 병합) → post_id 목록 반환
  → Frontend → GET /api/v1/upload/content/batch (상세 조회)

[텍스트 검색 요청]
  Frontend → GET /api/v1/search?query=고양이
  → SBERT 768d 인코딩
  → Redis KNN on "text_vector" → 순위 목록
  → (RRF 병합, 개인화 옵션) → post_id 목록 반환

[이미지 검색 요청]
  Frontend → POST /api/v1/discovery/image (이미지 파일)
  → CLIP 512d 인코딩
  → Redis KNN on "image_vector" → 순위 목록
  → (RRF 병합) → post_id 목록 반환
```
