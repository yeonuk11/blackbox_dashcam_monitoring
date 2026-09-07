"""Stage 2 — 충돌·진입 시점, 진입 방향, 회피 공간.

팀 계약(`predict(data_dir, model_dir) -> pandas.DataFrame`)과 이 폴더에 들어있는
기존 Stage 2 파이프라인 사이의 얇은 어댑터다. 실제 추론은 전부
``stage2_infer.predict_folder`` 안에서 일어나고, 여기서는 폴더 열거 · 프레임 번호
파싱 · 결과를 계약대로 정리하는 일만 한다.

왜 이런 구조인가
----------------
이 폴더의 ``stage2_*.py`` 는 리더보드에서 종합 0.65712(제출 80481)를 낸 번들과
**바이트 단위로 같은 파일**이다. 팀 패키지 규약에 맞추려고 그 안을 고쳐 쓰면
검증된 동작이 바뀔 위험이 있어서, 파일은 그대로 두고 이 어댑터만 새로 썼다.
바뀐 것은 import 다섯 줄뿐이다(``common`` → ``s2common``, ``model_io`` →
``s2model_io``). 다른 stage 도 같은 이름의 모듈을 번들할 수 있어서, 먼저 import 된
쪽이 이기는 ``sys.modules`` 충돌을 피하려고 접두사를 붙였다.

모듈들끼리는 ``import stage2_features`` 처럼 최상위 이름으로 서로를 부른다(대부분
함수 안에서 지연 import 한다). 그래서 아래에서 이 폴더를 ``sys.path`` 에 넣는다.
``stage2_`` 접두사가 붙어 있어 다른 stage 와 부딪히지 않는다.

체크포인트
----------
``model_dir`` 에 ``best.pt``(주 모델) 외에 ``v2_fold0.pt`` / ``v2_fold1.pt``(3-net
앙상블 파트너), ``stage2_backbone.pt``(파인튜닝 백본), ``convnext_tiny_in1k.pt``
(ImageNet 사전학습 가중치 — 평가 서버는 인터넷이 없어 직접 실어야 한다),
``ssdlite320_...pth``(회피공간 보조 검출기, 현재 기본 OFF)가 함께 있어야 한다.
전부 ``stage.json`` 에 등록돼 있고 SHA-256 으로 검증된다.

실패 정책
---------
한 폴더가 깨져도 전체 제출을 죽이지 않는다. ``stage2_infer`` 가 None 을 돌려주면
``s2common.s2_defaults`` 가 그 폴더 자신의 프레임 번호 범위 안에서 계산한 값을 쓴다
(존재하지 않는 프레임 번호를 내지 않는다). 어떤 경우에도 모든 ID 에 대해 한 행이
나가며, 값은 대회 규칙이 요구하는 범위·범주 안으로 강제된다.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import s2common as _c                                            # noqa: E402
import stage2_infer as _s2                                       # noqa: E402

COLUMNS = ["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"]


def _sample_dirs(data_dir: Path) -> list[Path]:
    """공식 ``<stage>/images`` 배치와 폴더를 바로 준 경우를 모두 받는다."""
    root = data_dir / "images" if (data_dir / "images").is_dir() else data_dir
    if not root.is_dir():
        raise FileNotFoundError(f"Stage 2 입력 폴더를 찾을 수 없습니다: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not dirs:
        raise FileNotFoundError(f"Stage 2 샘플 폴더가 없습니다: {root}")
    return dirs


def predict(data_dir: Path, model_dir: Path) -> pd.DataFrame:
    """``ID, collision_frame, entry_frame, evasion_space, entry_side`` 를 돌려준다."""
    data_dir, model_dir = Path(data_dir), Path(model_dir)
    _c.seed_everything()

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                             # noqa: BLE001
        device = "cpu"

    handle = _s2.load_model(model_dir, device=device)
    folders = _sample_dirs(data_dir)

    rows: list[dict] = []
    ids: list[str] = []
    n_frames: dict[str, int] = {}
    numbers_by_id: dict[str, list[int]] = {}

    for folder in folders:
        vid = folder.name
        frames = _c.list_frames(folder)          # 파일명 끝 숫자 기준 정렬
        numbers = [_c.frame_number(p) for p in frames]
        ids.append(vid)
        n_frames[vid] = len(frames)
        numbers_by_id[vid] = numbers
        if not frames:
            warnings.warn(f"Stage 2: 빈 폴더 {vid}; 기본값으로 채웁니다", RuntimeWarning)
            continue
        try:
            out = _s2.predict_folder(
                handle, str(folder),
                frames=[str(p) for p in frames],
                frame_numbers=numbers,
                mode="full",
            )
        except Exception as exc:                                  # noqa: BLE001
            # 한 폴더의 실패가 세 stage 전체를 오류로 만들지 않게 한다.
            warnings.warn(f"Stage 2 실패 {vid}: {type(exc).__name__}: {exc}", RuntimeWarning)
            out = None
        if out:
            rows.append({"ID": vid, **{k: out.get(k) for k in COLUMNS[1:]}})

    # 범위 클램프 + 범주 강제. 빠진 ID 는 그 폴더 자신의 프레임 번호로 채운다.
    # ``sanitize_stage2`` 가 이미 COLUMNS 순서·int64 dtype 의 DataFrame 을 돌려준다.
    df = _c.sanitize_stage2(rows, ids, n_frames, numbers_by_id)
    assert list(df.columns) == COLUMNS, f"열 계약 위반: {list(df.columns)}"
    return df.reset_index(drop=True)
