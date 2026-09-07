# blackbox_dashcam_monitoring

DACON **블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회**를 위한
3-Stage 팀 코드 저장소입니다. 이 저장소는 완성된 가중치를 배포하는 저장소가
아니라, 팀원이 `main`에서 stage별 브랜치를 만들고 코드 리뷰 후 병합하는 개발
저장소로 사용합니다.

## 현재 상태

| Stage | 목표 | 코드 상태 | 작업 브랜치 |
|---|---|---|---|
| 1 | 원본/재촬영 판별 | 도메인 적응 및 물리 재촬영 파인튜닝 완료 | `stage1/convnextv2-domain-finetune` |
| 2 | 사고 주요시점·상황 분석 | 담당자 구현 전 placeholder | `stage2/<작업명>` |
| 3 | 차량 거동 특성 분석 | 담당자 구현 전 placeholder | `stage3/<작업명>` |

`main`은 공용 제출 계약과 도구를 보관합니다. Stage 1의 실제 ConvNeXt V2-Nano
추론 코드와 결과 문서는 위 Stage 1 브랜치에서 관리하며, 검토 후 Pull Request로
`main`에 병합합니다.

## 협업 방식

```bash
git clone https://github.com/yeonuk11/blackbox_dashcam_monitoring.git
cd blackbox_dashcam_monitoring
git switch main
git pull --ff-only origin main
git switch -c stage2/baseline
```

각 담당자는 원칙적으로 `submission/model/stageN/`과 자신의 학습·실험 문서만
수정합니다. 공용 파일인 `submission/inference.py`, `submission/requirements.txt`,
`tools/`를 바꿔야 하면 다른 stage에 미치는 영향을 PR에 적고 팀 리뷰를 받습니다.
세부 규칙은 [팀 작업 규칙](docs/TEAM_WORKFLOW.md)과
[추론 인터페이스 계약](docs/INFERENCE_CONTRACT.md)을 따릅니다.

## 모델 가중치: Google Drive 사용

GitHub 일반 Git은 파일 하나가 100 MB를 넘으면 push를 거부합니다. 용량과 관계없이
`*.pt`, `*.pth`, `*.ckpt`는 Git에 커밋하지 않고 **Google Drive에 업로드**합니다.
각 stage 담당자는 공유 링크, 정확한 파일명, 바이트 크기, SHA-256을 자신의
`submission/model/stageN/stage.json`과 README에 기록해야 합니다.

최종 제출 담당자는 Drive에서 받은 파일을 다음 위치에 모읍니다.

```text
submission/model/stage1/best.pt
submission/model/stage2/best.pt
submission/model/stage3/best.pt
```

가중치는 `.gitignore`로 Git 추적에서 제외되지만, `build_submission.py`가 만드는
최종 ZIP에는 포함됩니다. 평가 서버는 인터넷이 차단되므로 Drive 다운로드는 반드시
ZIP 생성 **전에 로컬에서** 끝내야 합니다.

## DACON 제출 구조

```text
submission.zip
├── inference.py
├── requirements.txt
└── model/
    ├── stage1/
    ├── stage2/
    └── stage3/
```

이 대회에서는 평가 서버가 실행용 `script.py`를 자동으로 추가합니다. 따라서
제출 ZIP에는 `script.py`를 넣지 않습니다. `inference.py`는 최상위 함수
`predict_stage1`, `predict_stage2`, `predict_stage3`를 모두 제공해야 합니다.

## 검사와 제출 ZIP 생성

개발 중에는 가중치와 미구현 stage가 없어도 스켈레톤 검사를 할 수 있습니다.

```bash
python tools/check_repository.py
python tools/build_submission.py --allow-incomplete
python tools/validate_submission.py dist/submission.zip
```

세 stage 코드가 병합되고 Drive 가중치를 모두 배치한 뒤에는 예외 옵션 없이
최종 빌드를 실행합니다. 이 모드에서는 `stage.json`, 구현 완료 표시, 필수 가중치,
SHA-256이 모두 맞아야 ZIP이 생성됩니다.

```bash
python tools/verify_weights.py
python tools/build_submission.py
python tools/validate_submission.py --require-weights dist/submission.zip
```

현재 `main`만 내려받은 상태는 **즉시 제출 가능한 완성본이 아닙니다**. Stage 1~3
구현이 모두 병합되고 각 stage의 Drive 가중치를 정확한 경로에 배치한 뒤 위 최종
검사를 통과해야 제출할 수 있습니다.

## 공식 기준

- [대회 평가 및 코드 제출 규칙](https://www.dacon.io/competitions/official/236753/overview/evaluation)
- [Stage별 입력·출력 및 평가 안내](https://dacon.io/competitions/official/236753/talkboard/417186)
- [DACON 범용 코드 제출 가이드](https://cfiles.dacon.co.kr/competitions/236564/guide.html)

범용 가이드와 대회별 안내가 다르면 이 대회의 평가 페이지를 우선합니다.
