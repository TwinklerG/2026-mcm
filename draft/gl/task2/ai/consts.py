from pathlib import Path

DATA_RAW = Path(__file__).parent.parent.parent / "raw"
DATA_PROCESSED = Path(__file__).parent.parent.parent / "processed"
DATA_LINKED = Path(__file__).parent.parent.parent / "linked"
DATA_RES = Path(__file__).parent.parent.parent / "res"
DATA_RES.mkdir(exist_ok=True)
