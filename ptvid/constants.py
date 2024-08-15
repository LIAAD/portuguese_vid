import multiprocessing as mp
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

ROOT = Path().parent

CACHE_DIR = ROOT / "cache"
RAW_DATA_DIR = ROOT / "data" / "raw" / "data"
HF_DATA_DIR = ROOT / "data" / "hf"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

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

SAMPLE_SIZE = 3_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RAW_DATASET_NAME = "arubenruben/portuguese-language-identification-raw"
DATASET_NAME = "liaad/PtBrVId"

load_dotenv(ROOT)
HF_TOKEN = os.getenv("HF_TOKEN")


MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
