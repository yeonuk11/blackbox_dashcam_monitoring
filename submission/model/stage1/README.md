# Stage 1: ConvNeXt V2-Nano 재촬영 탐지

## 상태

- 블랙박스 원본 영상 도메인 적응 완료
- GoPro/Samsung × LG/노트북 물리 재촬영 paired 데이터 파인튜닝 완료
- 최종 체크포인트: epoch 16, validation 선택 임곗값 `0.37`
- 구현 진입점: `predict.py`의 `predict(data_dir, model_dir)`
- 출력: `ID`, `answer` (`ORIGINAL` 또는 `RERECORDED`)

## 모델과 추론

- ConvNeXt V2-Nano, 15,627,641 parameters
- 영상당 normalized temporal-bin center 8프레임
- 프레임 특징 640차원의 시간 평균과 population 표준편차 결합
- 1280×518 학습 입력에 맞춘 633×256 decode resize 후 224×224
  left/center/right 3-crop 확률 평균
- ImageNet mean/std 정규화
- 체크포인트와 디코더 모두 온라인 다운로드 없이 로컬에서만 사용

## 학습 결과

| 평가 세트 | Accuracy | Macro-F1 |
|---|---:|---:|
| paired validation (90개) | 1.000000 | 1.000000 |
| paired held-out test (90개) | 1.000000 | 1.000000 |
| DACON 제공 라벨 샘플 (10개) | 0.500000 | 0.450549 |

paired split은 원본/재촬영 장면 ID 누수를 막았지만, 같은 촬영 batch 안에 여러
split의 clip이 함께 들어 있어 장치·모니터·세션 특성이 공유됩니다. 따라서 paired
test 1.0은 낙관적일 수 있으며 공식 hidden-test 성능으로 해석하면 안 됩니다.
DACON 샘플도 10개뿐이므로 한두 예측에 점수가 크게 변합니다.

## 가중치

`best.pt`는 Git에 커밋하지 않고 Google Drive에 업로드해야 합니다.

- 로컬 파일명: `best.pt`
- 바이트: `62,573,431`
- SHA-256: `568e7831b271793d8c211c7a3eba3af2a6de22c2e6681e1de89cf2aa4d3492e8`
- Google Drive: **업로드 후 공유 링크를 `stage.json`과 이 문서에 입력할 것**

Drive에서 받은 파일을 이 README와 같은 폴더에 `best.pt`로 저장한 뒤 검증합니다.

```bash
python tools/verify_weights.py --allow-incomplete
```

`--allow-incomplete`은 아직 Stage 2/3가 끝나지 않은 현재 협업 단계에서만 사용합니다.
세 stage 통합 후에는 옵션 없이 실행해야 합니다.

## 주요 학습 설정

- seed 42, AdamW, weight decay 0.05
- batch 4 pairs, head LR 2e-4, backbone LR 1e-5
- 18 epochs, warm-up 1 epoch + cosine decay
- label-smoothed BCE + 0.20 × paired ranking loss (margin 0.50)
- epoch 1~3 head only, 4~8 head + stage 3, 9~18 head + stages 2~3
- AMP fp16, gradient clipping 1.0
- 합성 모아레, scanline, bezel, playback UI, JPEG 재압축, 원근 보정은 사용하지 않음

추론 중 디코딩할 수 없는 파일은 경고를 출력하고 보수적으로 `RERECORDED`로
처리합니다.
