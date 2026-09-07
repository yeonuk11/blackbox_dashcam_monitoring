# blackbox_dashcam_monitoring

DACON **블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회**의 3-Stage
팀 프로젝트 저장소입니다. `main`은 공용 제출 계약과 빌드 도구만 유지하고, 각
담당자는 stage별 브랜치에서 구현한 뒤 Pull Request로 병합합니다.

## 담당 범위

| Stage | 목표 | 작업 디렉터리 |
|---|---|---|
| 1 | 원본/재촬영 영상 판별 | `submission/model/stage1/` |
| 2 | 사고 주요시점·상황 분석 | `submission/model/stage2/` |
| 3 | 차량 거동 특성 분석 | `submission/model/stage3/` |

현재 stage 모듈은 API와 제출 구조를 검증하기 위한 placeholder입니다. 실제 대회
제출 전에는 세 stage 구현과 로컬 가중치를 모두 연결해야 합니다.

## 제출 구조

```text
submission.zip
├── inference.py
├── requirements.txt
└── model/
    ├── stage1/
    ├── stage2/
    └── stage3/
```

대회 평가 서버가 실행용 `script.py`를 추가하므로 ZIP 안에는 `script.py`를 넣지
않습니다. `inference.py`는 `predict_stage1`, `predict_stage2`, `predict_stage3`
함수를 모두 공개합니다.

## 빠른 검사

```bash
python tools/check_repository.py
python tools/build_submission.py --allow-missing-weights
python tools/validate_submission.py dist/submission.zip
```

실제 모델 제출본을 만들 때는 각 stage에 `best.pt`를 로컬로 배치한 뒤 다음을
실행합니다. `*.pt`는 `.gitignore` 대상이므로 ZIP에는 들어가지만 Git에는 올라가지
않습니다.

```bash
python tools/build_submission.py
```

팀 브랜치·병합 규칙과 체크포인트 공유 방법은
[`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md)를 참고하세요.
