from pathlib import Path

ROOT = Path().parent

RAW_DATA_DIR = ROOT / "data" / "raw" / "data"
HF_DATA_DIR = ROOT / "data" / "hf"

DOMAINS = ["journalistic", "literature", "legal", "politics", "web", "social_media"]

LABEL2ID = {
    "PT-PT": 0,
    "PT-BR": 1,
}

ID2LABEL = {
    0: "PT-PT",
    1: "PT-BR",
}
