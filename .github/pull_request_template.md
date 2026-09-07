## 변경 Stage

- [ ] Stage 1
- [ ] Stage 2
- [ ] Stage 3
- [ ] 공용 제출 코드/문서

## 변경 내용

<!-- 모델, 전처리, 출력 형식 변경을 요약하세요. -->

## 공용 파일 영향

- [ ] `inference.py`, `requirements.txt`, `tools/`를 변경하지 않음
- [ ] 공용 파일을 변경했다면 이유와 다른 stage 영향을 설명하고 팀 리뷰를 요청함

## 검증

- [ ] `python tools/check_repository.py`
- [ ] 담당 stage smoke test
- [ ] `stage.json`의 `implemented`, 파일명, 바이트, SHA-256, Drive 링크 갱신
- [ ] train/validation/test 누수 확인
- [ ] 대회 출력 열과 label 값 확인
- [ ] 가중치 SHA-256 기록
- [ ] 가중치를 Git이 아닌 Google Drive에 업로드하고 공유 권한 확인
- [ ] 100MB 이상 파일이 Git에 추적되지 않음
- [ ] 다른 stage API에 영향을 주지 않음
