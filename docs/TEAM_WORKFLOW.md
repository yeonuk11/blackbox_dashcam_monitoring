# 팀 작업 및 병합 규칙

## 브랜치와 책임 경계

- `main`: 공용 API, 제출 빌드/검증 도구, 리뷰가 끝난 병합본
- `stage1/<작업명>`: Stage 1 담당자
- `stage2/<작업명>`: Stage 2 담당자
- `stage3/<작업명>`: Stage 3 담당자

각 담당자는 자신의 `submission/model/stageN/`만 직접 수정하는 것을 원칙으로
합니다. `submission/inference.py`, `submission/requirements.txt`, `tools/`는 공동
소유 파일이므로 변경 이유, 출력 호환성, 다른 stage 영향, 검증 결과를 PR에
기록하고 최소 한 명의 팀 리뷰를 받은 뒤 병합합니다.

```bash
git switch main
git pull --ff-only origin main
git switch -c stage3/gru-baseline
```

## Stage 모듈 계약

각 `submission/model/stageN/__init__.py`는 아래 함수를 export합니다.

```python
def predict(data_dir, model_dir) -> pandas.DataFrame:
    ...
```

- 입력은 전달받은 `data_dir`, `model_dir`만 사용합니다.
- `/home/...` 같은 절대 경로, 인터넷 다운로드, test-time 학습을 금지합니다.
- 파일 순서를 고정하고 추론은 결정적으로 실행합니다.
- `model_dir / "best.pt"`를 기본 체크포인트로 사용합니다.
- 다른 stage의 내부 모듈을 import하지 않습니다.
- 공용 `inference.py`에는 모델 구현을 넣지 않고 adapter와 출력 검증만 둡니다.

정확한 열·라벨·중복 규칙은 [추론 인터페이스 계약](INFERENCE_CONTRACT.md)을
참고합니다.

## Stage 상태와 가중치

각 stage의 `stage.json`에는 다음을 기록합니다.

- 구현 완료 여부 `implemented`
- 필수 가중치 파일명
- Google Drive 공유 링크
- 파일 바이트 크기
- SHA-256

가중치는 Git에 커밋하지 않고 Google Drive에 업로드합니다. 링크 접근 권한은
"링크가 있는 사용자에게 보기 허용"으로 설정하고, 최종 제출 담당자가 Drive에서
내려받아 각 stage 폴더에 배치합니다. `implemented: false`이거나 체크섬이 다른
stage가 하나라도 있으면 최종 빌드는 실패합니다.

## 병합 순서

1. 담당자가 `stageN/*` 브랜치에서 코드·문서·`stage.json`을 완성합니다.
2. 가중치를 Drive에 올리고 링크와 SHA-256을 기록합니다.
3. 담당 stage smoke test와 저장소 검사를 수행합니다.
4. Pull Request를 열고 공용 파일 변경 여부를 명시합니다.
5. 리뷰 후 `main`에 병합합니다.
6. 최종 담당자가 세 가중치를 취합하고 제출 ZIP을 생성합니다.

## PR 전 검사

```bash
python tools/check_repository.py
python -m py_compile submission/inference.py submission/model/stage*/predict.py
python tools/build_submission.py --allow-incomplete
python tools/validate_submission.py dist/submission.zip
```

통합 완료 후에는 다음 최종 검사를 추가합니다.

```bash
python tools/verify_weights.py
python tools/build_submission.py
python tools/validate_submission.py --require-weights dist/submission.zip
```
