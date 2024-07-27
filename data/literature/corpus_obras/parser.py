from pathlib import Path
import dotenv
import os
from tqdm import tqdm
import re
from datasets import Dataset, Features, Value, ClassLabel


dotenv.load_dotenv(dotenv.find_dotenv())

CURRENT_PATH = Path(__file__).parent

DATA_PATH = os.path.join(CURRENT_PATH, "data")

CURRENT_STATE = "START"

with open(os.path.join(DATA_PATH, "corpus.txt"), "r", encoding="iso-8859-1") as f:
    
    corpora = {
        "text": [],
        "label": []
    }

    string = ""

    for line in tqdm(f.readlines()):
        
        if CURRENT_STATE == "OBRA":
            if string != "":
                corpora["text"].append(string)
                corpora["label"].append("pt-BR")
                print(f"Added new text:")              
            else:
                print("Warning: Empty string")

            string = ""

        elif CURRENT_STATE == "SENTENCE":
            line = line.replace("\n", " ").replace("\t", " ").strip()

            line = re.sub(r'<[^>]*>', ' ', line)

            string += line.split(" ")[0] + " "
        
        if '<obra' in line:
            CURRENT_STATE = "OBRA"

        elif '<s>' in line:
            CURRENT_STATE = "SENTENCE"

        elif '</obra>' in line:
            CURRENT_STATE = "START"


dataset = Dataset.from_dict(corpora, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).push_to_hub('corpus-obras', token=os.getenv('HF_TOKEN'))
