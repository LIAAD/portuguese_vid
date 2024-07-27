from datasets import load_dataset, Features, Value, ClassLabel, Dataset
import os
import pandas as pd
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())

dataset = load_dataset('rufimelo/PortugueseLegalSentences-v3', split='train')

dataset = dataset.select_columns(['text'])

df = pd.DataFrame(dataset['text'], columns=['text'])

df['label'] = 'pt-PT'

dataset = Dataset.from_pandas(df, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.2, shuffle=True, seed=42).push_to_hub(
    'portuguese_legal_sentences', token=os.getenv("HF_TOKEN"))
