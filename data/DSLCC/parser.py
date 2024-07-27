from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel, DatasetInfo
import os
import dotenv
import pandas as pd

CURRENT_PATH = Path(__file__).parent

SPLITER = "	"

FEATURES = Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
})

dotenv.load_dotenv(dotenv.find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")

dataset_dict = {
    "train": None,
    "validation": None,
    "test": None,
}

for split in ["train", "validation", "test"]:
    
    dataset = {
        "text": [],
        "label": []
    }

    for variant in ['pt-PT', 'pt-BR']:

        with open(os.path.join(CURRENT_PATH, 'data', f"{split}.txt"), "r", encoding="utf-8") as f:
            for line in f:
                text, label = line.split(SPLITER)

                text = text.strip()
                label = label.strip()

                if label != "pt-PT" and label != "pt-BR":
                    continue

                dataset["text"].append(text.strip())
                dataset["label"].append(label)

    dataset_dict[split] = Dataset.from_dict(
        dataset,
        features=FEATURES,
        split=split,
    )

dataset = DatasetDict(dataset_dict)

dataset.push_to_hub(f'portuguese_dslcc', token=HF_TOKEN)