from datasets import load_dataset, Dataset, Features, Value, ClassLabel
import dotenv
from tqdm import tqdm
import os

dotenv.load_dotenv(dotenv.find_dotenv())

dataset_dict = {
    "text": [],
    "label": []
}

dataset = load_dataset("europarl_bilingual", lang1="en",
                       lang2="pt", split="train", streaming=True)

counter = 0

for data in tqdm(dataset):
    dataset_dict["text"].append(data['translation']["pt"])
    dataset_dict["label"].append("pt-PT")

    counter += 1

    if counter % 100000 == 0 and counter > 0:
        dataset = Dataset.from_dict(dataset_dict, split="train", features=Features({
            "text": Value("string"),
            "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
        }))

        dataset.train_test_split(test_size=0.2, shuffle=True).push_to_hub(
            'portuguese_europarl', token=os.getenv("HF_TOKEN")
        )
