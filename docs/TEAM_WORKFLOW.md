# 팀 작업 및 병합 규칙

## 브랜치

- `main`: 공용 API, 제출 빌드 도구, 병합 완료본만 유지
- `stage1/<작업명>`: Stage 1 담당자
- `stage2/<작업명>`: Stage 2 담당자
- `stage3/<작업명>`: Stage 3 담당자

예시:

```bash
git switch main
git pull --ff-only origin main
git switch -c stage2/baseline
```

각 담당자는 원칙적으로 자신의 `submission/model/stageN/`과 관련 문서만 수정합니다.
공용 `submission/inference.py` 또는 `requirements.txt` 변경이 필요하면 PR 설명에
변경 이유와 다른 stage에 미치는 영향을 적습니다.

## Stage 모듈 계약

각 `submission/model/stageN/__init__.py`는 다음 함수를 export해야 합니다.

```python
def predict(data_dir, model_dir) -> pandas.DataFrame:
    ...
```

공용 `submission/inference.py`가 이 함수를 대회의 `predict_stageN` 이름으로
연결합니다. 절대 경로(`/home/...`)를 코드에 넣지 말고, 전달받은 `data_dir`과
`model_dir`만 사용합니다.

필수 출력 열:

- Stage 1: `ID`, `answer`
- Stage 2: `ID`, `collision_frame`, `entry_frame`, `evasion_space`, `entry_side`
- Stage 3: `ID`, `sample_index`, `accel_label`, `steer_label`

## 가중치와 대용량 파일

GitHub 일반 Git의 단일 파일 제한을 피하기 위해 모델 가중치, 데이터, 실험 결과,
ZIP은 기본적으로 Git에 커밋하지 않습니다. 각 stage의 실제 제출 가중치는 로컬에서
다음 경로에 둡니다.

```text
submission/model/stage1/best.pt
submission/model/stage2/best.pt
submission/model/stage3/best.pt
```

가중치는 팀 공유 스토리지 또는 GitHub Release asset으로 전달하고 SHA-256을 PR과
stage README에 기록합니다. Git LFS를 사용하려면 팀 전체가 설치·용량 정책에 먼저
동의한 뒤 별도의 설정 PR로 도입합니다. `.gitattributes`에 이름만 적는 것으로는
Git LFS가 활성화되지 않습니다.

## PR 전 검사

```bash
python tools/check_repository.py
python -m py_compile submission/inference.py submission/model/stage*/predict.py
python tools/build_submission.py --allow-missing-weights
python tools/validate_submission.py dist/submission.zip
```

실제 세 가중치를 갖춘 통합 환경에서는 `--allow-missing-weights` 없이 빌드합니다.
