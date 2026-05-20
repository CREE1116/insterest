# Insterest API Specification

마이크로서비스별 API 명세입니다. 모든 요청은 Nginx Ingress(`http://localhost`)를 통해 라우팅되며, 서비스 내부 포트는 참고용입니다.

---

## 1. Auth Service (내부 포트 8000)

사용자 인증 및 계정 관리를 담당합니다.

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 회원가입 | `POST` | `/api/v1/auth/register` | `{ email, password, nickname }` |
| 로그인 | `POST` | `/api/v1/auth/login` | Form Data(`username`, `password`). 성공 시 `access_token` 쿠키 설정 |
| 로그아웃 | `POST` | `/api/v1/auth/logout` | 인증 쿠키 삭제 |
| 내 정보 조회 | `GET` | `/api/v1/users/me` | 현재 로그인 사용자 기본 정보 |
| 사용자 일괄 조회 | `GET` | `/api/v1/users/batch` | `?user_ids=uuid1,uuid2` 닉네임·프로필 일괄 반환 |

---

## 2. Upload Service (내부 포트 8001)

미디어 업로드, 게시물 관리, 피드 데이터 제공을 담당합니다.

### 미디어

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 단일 업로드 | `POST` | `/api/v1/upload/media/upload` | `multipart/form-data`. `media_id`, `content_id` 반환 |
| 복합 업로드 | `POST` | `/api/v1/upload/media/upload-combined` | 이미지+오디오 동시 업로드. 하나의 `content_id`로 묶음 |
| 정적 파일 | `GET` | `/uploads/{filename}` | 업로드된 파일 직접 접근 |

### 게시물

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 게시물 생성 | `POST` | `/api/v1/upload/content/` | `content_id` 또는 `metadata_info`로 피드에 게시 |
| 전체 피드 | `GET` | `/api/v1/upload/content/feed` | 공개 게시물 목록. `?skip=0&limit=20` |
| 게시물 일괄 조회 | `POST` | `/api/v1/upload/content/batch` | `{ post_ids: [uuid, ...] }` 상세 정보 일괄 반환. 추천 결과 조회용 |
| 유저 게시물 | `GET` | `/api/v1/upload/content/users/{user_id}/posts` | 특정 사용자 게시물 목록 |
| 게시물 상세 | `GET` | `/api/v1/upload/content/{post_id}` | 미디어·좋아요 상태 포함 상세 반환 |
| 게시물 삭제 | `DELETE` | `/api/v1/upload/content/{post_id}` | 작성자 본인만 가능 |
| 전체 게시물 (내부) | `GET` | `/api/v1/upload/content/posts/all` | 추천 엔진 백필용 — 모든 게시물 |

---

## 3. User Service (내부 포트 8006)

프로필 및 컬렉션(저장함) 관리를 담당합니다.

### 프로필

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 프로필 조회 | `GET` | `/api/v1/users/me` | 자기소개 등 확장 프로필 |
| 프로필 수정 | `PUT` | `/api/v1/users/me` | 닉네임, 자기소개 수정 |
| 프로필 이미지 | `POST` | `/api/v1/users/me/image` | 프로필 사진 업로드 |

### 컬렉션

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 컬렉션 생성 | `POST` | `/api/v1/collections/` | 새 저장 폴더 생성 |
| 목록 조회 | `GET` | `/api/v1/collections/` | 내 전체 컬렉션 목록 |
| 아이템 추가 | `POST` | `/api/v1/collections/{id}/items` | `{ post_id }` |
| 아이템 목록 | `GET` | `/api/v1/collections/{id}/items` | 컬렉션 내 게시물 목록 |
| 아이템 삭제 | `DELETE` | `/api/v1/collections/items/{post_id}` | 내 모든 컬렉션에서 제거 |

---

## 4. Interaction Service (내부 포트 8004)

좋아요, 조회수 등 상호작용을 담당합니다.

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 좋아요 토글 | `POST` | `/api/v1/interactions/{post_id}/like` | 좋아요 / 취소 토글 |
| 저장 토글 | `POST` | `/api/v1/interactions/{post_id}/save` | 보관함 저장 / 취소 |
| 조회수 기록 | `POST` | `/api/v1/interactions/{post_id}/view` | 게시물 조회 시 호출 |
| 시청 시간 | `POST` | `/api/v1/interactions/watch-time` | `{ post_id, seconds }` |
| 게시물 통계 | `GET` | `/api/v1/interactions/stats/{post_id}` | 좋아요·저장·조회수·평균 시청 시간 |
| 내 좋아요 목록 | `GET` | `/api/v1/interactions/me/liked` | 내가 좋아요 누른 게시물 ID 목록 |
| 유저 활동 이력 | `GET` | `/api/v1/interactions/user/{user_id}` | 특정 유저 좋아요/저장 이력 (추천 엔진용) |
| 전체 이력 (내부) | `GET` | `/api/v1/interactions/all` | 전체 상호작용 데이터 |

---

## 5. Comment Service (내부 포트 8005)

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 댓글 작성 | `POST` | `/api/v1/comments/` | `{ post_id, content }` |
| 댓글 목록 | `GET` | `/api/v1/comments/{post_id}` | 특정 게시물의 전체 댓글 |
| 댓글 일괄 조회 | `POST` | `/api/v1/comments/batch` | 여러 게시물 댓글 일괄 반환 |

---

## 6. Recommendation Service (내부 포트 8008)

CLIP + SBERT 기반 추천·검색 엔진입니다. `/discovery`, `/search`, `/intel` 세 접두어를 사용합니다.

> `/search/*`는 `/discovery/*`와 동일한 라우터를 공유합니다.

### 피드 및 검색 (`/discovery`, `/search`)

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 개인화 피드 | `GET` | `/api/v1/discovery/recommend` | `?user_id=&query=&skip=&limit=`. RRF(SBERT 텍스트 + UserTower CLIP) 추천 결과를 post_id 배열로 반환 |
| 텍스트 검색 | `GET` | `/api/v1/search` | `?query=고양이&user_id=&skip=&limit=`. SBERT 순수 텍스트 검색 (개인화 없음) |
| 이미지 검색 | `POST` | `/api/v1/discovery/image` | `multipart/form-data` 이미지 파일. CLIP 이미지 임베딩 기반 유사 포스트 검색 |
| 학습 트리거 | `POST` | `/api/v1/discovery/train` | UserTower 수동 재학습 시작 (백그라운드 실행) |
| 벡터 백필 | `POST` | `/api/v1/discovery/sync` | DB 전체 포스트를 CLIP으로 재인코딩하여 Redis 갱신 |
| 오프라인 평가 | `GET` | `/api/v1/discovery/metrics` | NDCG·Recall 등 정량 지표 반환 |

### 벤치마크 (`/discovery/benchmark`)

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 합성 벤치마크 | `GET` | `/api/v1/discovery/benchmark/synthetic` | Pillow 도형·색상 10종으로 CLIP Retrieval 측정 |
| 커스텀 벤치마크 | `POST` | `/api/v1/discovery/benchmark/custom` | `dataset.json` + 이미지 ZIP 업로드. 커스텀 Recall 측정 |
| CIFAR-100 벤치마크 | `GET` | `/api/v1/discovery/benchmark/animal` | 100클래스 × 2샘플 실 이미지로 텍스트·이미지 교차 검색 Recall@1/3/5 평가. 데이터 영구 보존 |
| Gemini 상호작용 시드 | `POST` | `/api/v1/discovery/benchmark/seed-gemini` | `?force=false`. Gemini API로 50개 가상 페르소나 생성 및 좋아요 주입. 이미 존재하면 스킵 |

### 디버그

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 활동 유저 목록 | `GET` | `/api/v1/discovery/debug/users` | 좋아요 누적 상위 10명 (시뮬레이터용) |

### 엔진 관리 (`/intel`)

| 기능 | Method | Endpoint | 설명 |
|:---|:---:|:---|:---|
| 모델 학습 | `POST` | `/api/v1/intel/train` | UserTower 재학습 트리거 (비동기) |
| 전체 재인덱싱 | `POST` | `/api/v1/intel/backfill` | 모든 포스트 CLIP 재인코딩 및 Redis 갱신 |
| 엔진 상태 | `GET` | `/api/v1/intel/status` | `{ redis_vector_count, engine, status }` |
| 오프라인 평가 | `GET` | `/api/v1/intel` | NDCG·Recall 정량 지표 (`/discovery/metrics`와 동일) |

---

## 7. Generation Service (내부 포트 8002)

AI 이미지(Flux) 및 음악(MusicGen) 생성을 담당합니다. WebSocket 기반입니다.

**URL:** `ws://{host}/api/v1/ws/generate`

**인증:** 쿠키의 `access_token` 필요

**메시지 흐름:**
1. Client → Server: `{ "user_input": "prompt" }`
2. Server → Client: `{ "status": "loading" }`
3. Server → Client: `{ "status": "generating" }`
4. Server → Client: `{ "status": "completed", "data": { "image_url": "...", "audio_url": "..." } }`

---

## 공통

### 인증
쿠키 기반 JWT. 보호된 엔드포인트는 요청 헤더에 `access_token` 쿠키가 있어야 합니다.

### 정적 리소스 경로

| 리소스 | 경로 |
|:---|:---|
| 사용자 업로드 이미지/오디오 | `/uploads/{filename}` |
| AI 생성 결과물 | `/outputs/{filename}` |
| 프로필 사진 | `/profiles/{filename}` |

### 추천 결과 조회 패턴

추천 엔드포인트(`/discovery/recommend`, `/search`)는 `UUID[]`만 반환합니다. 실제 포스트 데이터는 Upload Service에서 일괄 조회합니다.

```
GET /api/v1/discovery/recommend?user_id=...
→ [uuid1, uuid2, uuid3, ...]

POST /api/v1/upload/content/batch
Body: { "post_ids": [uuid1, uuid2, uuid3, ...] }
→ [{ id, caption, media_list, like_count, ... }, ...]
```
