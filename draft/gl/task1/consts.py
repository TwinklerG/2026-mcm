from pathlib import Path

DATA_RAW = Path(__file__).parent.parent / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "processed"
DATA_PROCESSED.mkdir(exist_ok=True)
