import pandas as pd
from datasets import Dataset, DatasetDict

from ptvid.constants import HF_DATA_DIR, RAW_DATA_DIR

if __name__ == "__main__":
    pt_train = pd.read_csv(RAW_DATA_DIR / "dsl-tl" / "PT_train.tsv", sep="\t", header=None)
    pt_dev = pd.read_csv(RAW_DATA_DIR / "dsl-tl" / "PT_dev.tsv", sep="\t", header=None)

    # force column names to be: id, text, label
    pt_train.columns = ["id", "text", "label"]
    pt_dev.columns = ["id", "text", "label"]

    # Index column id
    pt_train = pt_train.set_index("id")
    pt_dev = pt_dev.set_index("id")

    # PT-BR: 1, PT-PT: 0, PT: 2
    pt_train["label"] = pt_train["label"].map({"PT-BR": 1, "PT-PT": 0, "PT": 2})
    pt_dev["label"] = pt_dev["label"].map({"PT-BR": 1, "PT-PT": 0, "PT": 2})

    # Drop the index column
    pt_train = pt_train.reset_index(drop=True)
    pt_dev = pt_dev.reset_index(drop=True)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(pt_train).shuffle(seed=42),
            "validation": Dataset.from_pandas(pt_dev).shuffle(seed=42),
        }
    )

    dataset.save_to_disk(HF_DATA_DIR / "dsl-tl")
