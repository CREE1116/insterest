# Interest Project API Specification (v2.0)

이 문서는 Interest 프로젝트의 마이크로서비스별 최신 API 명세서입니다. 모든 API는 `/api/v1` 접두어를 기본으로 합니다.

---

## 1. Auth Service (8000)
사용자 인증 및 계정 관리를 담당합니다. (`auth`, `social`, `users` 도메인 통합)

| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **회원가입** | `POST` | `/api/v1/auth/register` | 이메일, 비밀번호, 이름을 받아 가입 |
| **로그인** | `POST` | `/api/v1/auth/login` | Form Data(`username`, `password`) 로그인. 쿠키(`access_token`) 설정 |
| **로그아웃** | `POST` | `/api/v1/auth/logout` | 인증 쿠키 삭제 |
| **내 정보 조회** | `GET` | `/api/v1/users/me` | 현재 로그인된 사용자의 기본 정보 반환 |
| **사용자 일괄 조회** | `GET` | `/api/v1/users/batch` | `user_ids` 목록을 쿼리로 받아 닉네임/프로필 정보 반환 |

---

## 2. Upload Service (8001)
미디어 업로드, 보관함 관리, 피드 게시를 담당합니다. 모든 경로는 `/upload` 접두어를 가집니다.

### 미디어 관리
| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **미디어 업로드** | `POST` | `/api/v1/upload/media/upload` | 단일 파일 업로드. `media_id`, `content_id` 반환 |
| **복합 업로드** | `POST` | `/api/v1/upload/media/upload-combined` | 이미지+오디오 동시 업로드. 하나의 `content_id`로 묶임 |
| **정적 파일 접근** | `GET` | `/uploads/{filename}` | 업로드된 실제 이미지/오디오 파일 접근 (실제 파일 경로) |

### 콘텐츠 및 피드
| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **게시물 생성** | `POST` | `/api/v1/upload/content/` | `content_id` 또는 `metadata_info`를 사용하여 피드에 게시 |
| **전체 피드 조회** | `GET` | `/api/v1/upload/content/feed` | 전체 공개된 게시물 목록 조회 (Post 기준, 페이징 지원) |
| **게시물 일괄 조회** | `POST` | `/api/v1/upload/content/batch` | 여러 `post_ids`에 대해 상세 정보 일괄 반환 (추천 엔진 결과 조회용) |
| **특정 유저 게시물** | `GET` | `/api/v1/upload/content/users/{user_id}/posts` | 특정 사용자의 게시물 목록 조회 |
| **게시물 상세 조회** | `GET` | `/api/v1/upload/content/{post_id}` | 특정 게시물 상세 정보 및 좋아요 상태 조회 |
| **게시물 삭제** | `DELETE` | `/api/v1/upload/content/{post_id}` | 특정 게시물 삭제 (작성자 확인) |
| **전체 게시물 (내부용)** | `GET` | `/api/v1/upload/content/posts/all` | 추천 엔진 백필을 위한 모든 게시물 조회 |

---

## 3. User Service (8006)
프로필 고도화 및 개인화된 보관함(Collection) 기능을 담당합니다.

### 프로필 관리
| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **프로필 조회** | `GET` | `/api/v1/users/me` | 확장된 프로필 정보(자기소개 등) 조회 |
| **프로필 수정** | `PUT` | `/api/v1/users/me` | 닉네임, 자기소개 수정 |
| **프로필 이미지** | `POST` | `/api/v1/users/me/image` | 프로필 사진 업로드 |

### 컬렉션(저장함)
| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **컬렉션 생성** | `POST` | `/api/v1/collections/` | 새로운 이름의 저장 폴더 생성 |
| **목록 조회** | `GET` | `/api/v1/collections/` | 나의 전체 컬렉션 목록 조회 |
| **아이템 추가** | `POST` | `/api/v1/collections/{id}/items` | 특정 컬렉션에 게시물(`post_id`) 추가 |
| **아이템 조회** | `GET` | `/api/v1/collections/{id}/items` | 특정 컬렉션 내 게시물 목록 |
| **아이템 삭제** | `DELETE` | `/api/v1/collections/items/{post_id}` | 내 모든 컬렉션에서 해당 게시물 제거 |

---

## 4. Interaction Service (8004)
좋아요, 조회수 등 사용자 상호작용을 담당합니다.

| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **좋아요 토글** | `POST` | `/api/v1/interactions/{post_id}/like` | 좋아요 실행 또는 취소 |
| **저장 토글** | `POST` | `/api/v1/interactions/{post_id}/save` | 보관함 저장 실행 또는 취소 |
| **조회수 기록** | `POST` | `/api/v1/interactions/{post_id}/view` | 게시물 조회 시 호출 |
| **시청 시간 기록** | `POST` | `/api/v1/interactions/watch-time` | 시청 시간(초) 기록 |
| **게시물 통계** | `GET` | `/api/v1/interactions/stats/{post_id}` | 좋아요, 저장, 조회수, 평균 시청 시간 등 반환 |
| **내 좋아요 목록** | `GET` | `/api/v1/interactions/me/liked` | 내가 좋아요 누른 게시물 ID 목록 |
| **유저 활동 이력** | `GET` | `/api/v1/interactions/user/{user_id}` | 특정 유저의 좋아요/저장 이력 조회 (추천 엔진용) |
| **전체 활동 이력** | `GET` | `/api/v1/interactions/all` | 시스템의 모든 상호작용 데이터 (내부 분석용) |

---

## 5. Comment Service (8005)
게시물별 댓글 관리를 담당합니다.

| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **댓글 작성** | `POST` | `/api/v1/comments/` | 댓글 내용과 `post_id`를 전달 |
| **댓글 목록** | `GET` | `/api/v1/comments/{post_id}` | 특정 게시물의 댓글 전체 조회 |
| **댓글 일괄 조회** | `POST` | `/api/v1/comments/batch` | 여러 게시물 ID에 대해 댓글 목록 일괄 반환 |

---

## 6. Recommendation Service (8008)
AI 기반 추천 및 검색 엔진을 담당합니다. `/discovery`, `/search`, `/intel` 세 가지 접두어를 사용합니다.

### 추천 및 검색 (`/discovery`, `/search`)
| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **추천 피드** | `GET` | `/api/v1/discovery/recommend` | 개인화된 추천 게시물 ID 목록 반환 |
| **통합 검색** | `GET` | `/api/v1/search` | 검색어(`query`)와 유저 취향을 결합한 검색 결과 반환 |
| **추천 모델 학습** | `POST` | `/api/v1/discovery/train` | 추천 모델 수동 재학습 트리거 |
| **인덱스 동기화** | `POST` | `/api/v1/discovery/sync` | 벡터 데이터베이스 수동 갱신 (Backfill) |

### 지능형 상태 관리 (`/intel`)
| 기능 | Method | Endpoint | 설명 |
| :--- | :--- | :--- | :--- |
| **모델 학습** | `POST` | `/api/v1/intel/train` | 추천 모델 학습 트리거 (비동기) |
| **벡터 백필** | `POST` | `/api/v1/intel/backfill` | 전체 게시물 리인덱싱 트리거 |
| **엔진 상태** | `GET` | `/api/v1/intel/status` | 추천 엔진의 현재 상태 및 마지막 학습 정보 |
| **벤치마크** | `GET` | `/api/v1/intel/benchmark` | 정량적 지표(Recall, NDCG) 측정 결과 반환 |

---

## 7. Generation Service (8002)
AI 콘텐츠(이미지, 음악) 생성을 담당합니다.

### WebSocket 생성
- **URL:** `ws://{host}/api/v1/ws/generate`
- **인증:** 쿠키의 `access_token` 필요
- **메시지 흐름:**
  1. Client -> Server: `{"user_input": "prompt"}`
  2. Server -> Client: `{"status": "loading", ...}`
  3. Server -> Client: `{"status": "generating", ...}`
  4. Server -> Client: `{"status": "completed", "data": {"image_url": "...", "audio_url": "...", ...}}`

---

## 공통 인프라
- **정적 리소스:**
  - 사용 프로필: `/profiles/{filename}`
  - 사용자 업로드: `/uploads/{filename}`
  - AI 생성물: `/outputs/{filename}`
