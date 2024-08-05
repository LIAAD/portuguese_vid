import multiprocessing as mp

from pathlib import Path

ROOT = Path().parent

RAW_DATA_DIR = ROOT / "data" / "raw" / "data"
HF_DATA_DIR = ROOT / "data" / "hf"

N_PROC = mp.cpu_count()

DOMAINS = ["journalistic", "literature", "legal", "politics", "web", "social_media"]

LABEL2ID = {
    "PT-PT": 0,
    "PT-BR": 1,
}

ID2LABEL = {
    0: "PT-PT",
    1: "PT-BR",
}


SAMPLE_SIZE = 1_000
