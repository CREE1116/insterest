---
name: orchestration-management
description: >-
  Orchestrator agent coordination skill covering agent role boundaries, delegation
  patterns, escalation paths, multi-agent pipelines, and result synthesis.
  Defines which agent owns what and how they interact.
---

# Orchestration Management

이 스킬 문서는 Insterest 프로젝트의 **오케스트레이터 에이전트(Orchestrator Agent, OA)**가 7개 서브에이전트를 조율하는 방식과 원칙을 정의합니다.

---

## 1. 에이전트 역할 경계 (Role Boundaries)

각 에이전트는 명확히 분리된 책임 영역을 가집니다. **역할 혼합(Role Mixing)은 절대 금지**입니다.

| 에이전트 | 유형 | 책임 | 금지 사항 |
|---|---|---|---|
| **Platform Reliability Manager** | 진단·운영 | K8s, 인프라, 배포 안정성 | 코드 수정 |
| **Reliability & Observability Manager** | 진단·계측 | 메트릭, 트레이싱, SLO, 장애 탐지 | 코드 수정 |
| **Backend Architecture Manager** | 설계·감사 | API 설계, 도메인 모델, 트랜잭션 경계 | 직접 배포 |
| **Recommendation Intelligence Manager** | 분석·제안 | ML 품질 감사, 드리프트 탐지, 개선 방향 제시 | 코드 직접 수정 |
| **Safe Change Guardian** | 검토·방어 | 변경 리뷰, 하위 호환성, Blast Radius 검증 | 새 기능 개발 |
| **Security & Access Manager** | 감사·방어 | 인증, 시크릿, OWASP 취약점 | 직접 배포 |
| **Product Flow Auditor** | 감사·제안 | UX-BE 계약, 페이지네이션, 프론트 흐름 | 코드 직접 수정 |
| **Code Refactoring Agent** | 구현·배포 | 코드 수정, 브랜치, PR, CI 루프 | 설계 결정 |

---

## 2. 위임 패턴 (Delegation Patterns)

### A. 분석 → 구현 2단계 파이프라인 (가장 중요)

분석 에이전트가 먼저 감사하고 계획을 수립합니다. 구현은 반드시 **Code Refactoring Agent**가 담당합니다.

```
오케스트레이터
    │
    ├─[1단계]→ 분석 에이전트 (RIM / BAM / SCG / SAM / PFA)
    │               ↓ 감사 결과 + 구체적 구현 명세 보고
    │
    ├─[2단계]→ Code Refactoring Agent
    │               ↓ 코드 수정 → 브랜치 → PR → CI 루프
    │
    └─[병합]→ 오케스트레이터가 CI 그린 확인 후 병합 결정
```

**잘못된 패턴** (절대 하지 말 것):
- 분석 에이전트에게 코드 수정까지 지시
- Code Refactoring Agent에게 아키텍처 설계를 맡기는 것
- 오케스트레이터가 직접 코드를 수정하는 것

### B. 장애 대응 병렬 에스컬레이션

장애 발생 시 진단 에이전트들을 병렬로 실행합니다.

```
오케스트레이터
    ├─→ Reliability & Observability Manager  (메트릭/트레이싱 이상 탐지)
    ├─→ Platform Reliability Manager         (K8s/인프라 상태 확인)
    └─→ Recommendation Intelligence Manager  (ML/ANN 레이어 진단)
            ↓ 모두 보고 완료 후
    오케스트레이터가 근본 원인 종합 → Code Refactoring Agent에 수정 위임
```

### C. 코드 변경 거버넌스 파이프라인

모든 기능 개발·리팩토링 전에 반드시 거버넌스 에이전트를 경유합니다.

```
오케스트레이터
    ├─[선제 검토]→ Safe Change Guardian (변경 범위 / Blast Radius 검토)
    ├─[선제 검토]→ Security & Access Manager (보안 영향 검토) [필요 시]
    │                   ↓ 승인 또는 수정 요청
    └─→ Code Refactoring Agent (승인된 계획만 구현)
```

---

## 3. 보고 프로토콜 (Reporting Protocol)

서브에이전트는 작업 완료 후 반드시 `send_message`로 오케스트레이터에게 보고합니다.

### 분석 에이전트 보고 형식
```
[에이전트 명] 분석 완료 보고
- 감사 대상: ...
- 발견된 문제: ...
- 제안 개선 방향: (구체적 파일명, 함수명, 변경 내용 포함)
- 우선순위: High / Medium / Low
```

### 구현 에이전트 (Code Refactoring Agent) 보고 형식
```
[Code Refactoring Agent] 구현 완료 보고
- PR URL: ...
- CI 상태: ✅ All Green / ❌ Failing
- 변경 요약: ...
- 병합 준비: Yes / No
```

---

## 4. 오케스트레이터 의사결정 원칙

1. **역할 위임 우선**: 오케스트레이터는 직접 코드를 수정하거나 진단 작업을 수행하지 않습니다. 항상 적절한 서브에이전트에 위임합니다.

2. **순서 준수**: 분석 → 검토 → 구현 → CI 검증 → 병합 순서를 반드시 지킵니다. 분석 없이 바로 구현을 지시하지 않습니다.

3. **병렬 실행 활용**: 독립적인 진단 작업(예: 인프라 + ML + 보안 동시 감사)은 병렬로 서브에이전트를 실행하여 속도를 높입니다.

4. **병합 최종 승인**: PR 병합은 반드시 오케스트레이터가 CI 결과를 확인한 후 결정합니다. Code Refactoring Agent가 자율적으로 병합하지 않습니다.

5. **컨텍스트 전달 책임**: 오케스트레이터는 서브에이전트에게 작업을 위임할 때 충분한 컨텍스트(관련 파일 경로, 이전 분석 결과, 제약 조건)를 함께 전달합니다.

---

## 5. 에이전트 간 인터페이스 예시

### 예시 1: 추천 시스템 성능 개선 요청

```
[올바른 흐름]

오케스트레이터
 └─[1]→ Recommendation Intelligence Manager
         - 현재 ML 파이프라인 감사
         - 임베딩 드리프트, ANN 성능, 다양성 분석
         - 구체적 개선 명세 작성 (파일, 함수 단위)
         → 오케스트레이터에게 보고

 └─[2]→ Safe Change Guardian
         - RIM 제안의 Blast Radius 검토
         - 하위 호환성 영향 분석
         → 오케스트레이터에게 승인/수정 보고

 └─[3]→ Code Refactoring Agent
         - 승인된 명세를 코드로 구현
         - 브랜치 → PR → CI 루프
         → 오케스트레이터에게 PR URL + CI 상태 보고

 └─[4] 오케스트레이터: CI 그린 확인 → 병합 실행
```

### 예시 2: 장애 대응

```
[올바른 흐름]

오케스트레이터 (장애 감지)
 ├─[병렬]→ Reliability & Observability Manager (메트릭 이상 탐지)
 ├─[병렬]→ Platform Reliability Manager (Pod/노드 상태 확인)
 └─[병렬]→ Recommendation Intelligence Manager (ML 레이어 진단)

 → 모든 보고 수집 후 오케스트레이터가 근본 원인 종합
 → Code Refactoring Agent에 핫픽스 위임 (SCG 검토 병행)
```
