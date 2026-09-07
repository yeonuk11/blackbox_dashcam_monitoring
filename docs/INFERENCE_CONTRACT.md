# 공용 inference 계약

이 문서는 세 담당자의 코드를 하나의 DACON 제출본으로 합칠 때 지켜야 할 고정
계약입니다. 모델 구조는 stage별로 자유롭게 바꿀 수 있지만 이 인터페이스는 PR
합의 없이 바꾸지 않습니다.

## 공용 진입점

`submission/inference.py`에는 정확히 다음 세 공개 함수가 있어야 합니다.

```python
predict_stage1(data_dir, model_dir) -> pandas.DataFrame
predict_stage2(data_dir, model_dir) -> pandas.DataFrame
predict_stage3(data_dir, model_dir) -> pandas.DataFrame
```

공용 함수는 `model.stageN.predict`를 호출한 뒤 결과 계약을 검증합니다. 실제 모델,
디코더, 전처리, 체크포인트 로드는 `submission/model/stageN/` 안에 둡니다.

## 출력 계약

| Stage | 정확한 열 순서 | 추가 조건 |
|---|---|---|
| 1 | `ID`, `answer` | `answer`는 `ORIGINAL` 또는 `RERECORDED`, ID 중복 금지 |
| 2 | `ID`, `collision_frame`, `entry_frame`, `evasion_space`, `entry_side` | ID 중복 금지 |
| 3 | `ID`, `sample_index`, `accel_label`, `steer_label` | `(ID, sample_index)` 중복 금지 |

Stage 3의 `accel_label`은 `STOPPED`, `ACCELERATING`, `DECELERATING`,
`CONSTANT`, `steer_label`은 `LEFT`, `STRAIGHT`, `RIGHT`만 허용합니다. 열이
누락되거나 순서가 다르거나 알 수 없는 라벨이 나오면 공용 wrapper가 즉시 오류를
발생시켜 잘못된 제출을 조기에 막습니다.

## 구현 규칙

- 입력 ID는 영상 파일명 stem을 기준으로 하며 처리 순서를 정렬합니다.
- 숨겨진 테스트의 해상도·길이·FPS가 예시와 다를 수 있으므로 이를 상수로 가정해
  파일을 거르지 않습니다.
- 체크포인트는 제출 ZIP 내부의 `model_dir`에서만 읽고 온라인 다운로드를 하지
  않습니다. pretrained 생성자는 `weights=None` 또는 동등한 오프라인 설정을
  사용합니다.
- 다른 샘플의 통계나 ID 규칙으로 답을 보정하지 않으며 test-time 학습,
  pseudo-labeling을 하지 않습니다.
- Stage 2의 시점은 리사이즈/샘플링 인덱스가 아닌 원본 영상 프레임 번호로
  복원합니다.
- Stage 3는 요구되는 모든 `sample_index` 행을 반환합니다.
- CUDA가 없을 때도 import 자체는 성공해야 하며, 로컬 절대 경로를 넣지 않습니다.
- 디코딩 실패 정책은 stage README에 명시하고, 무조건 예외를 숨기는 구현은 피합니다.

## 공용 파일 변경 절차

`inference.py`나 `requirements.txt`를 변경하는 PR은 다음을 포함해야 합니다.

1. 변경이 필요한 이유와 영향받는 stage
2. 세 함수 import 및 호출 smoke test
3. 정확한 출력 열과 라벨 검증 결과
4. 오프라인 실행 및 패키지 설치 시간 영향
5. 제출 ZIP 구조 검사 결과

공식 기준은 [대회 평가 페이지](https://www.dacon.io/competitions/official/236753/overview/evaluation)와
[Stage 안내](https://dacon.io/competitions/official/236753/talkboard/417186)를 따릅니다.
