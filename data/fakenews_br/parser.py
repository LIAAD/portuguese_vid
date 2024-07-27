import pandas as pd
from pathlib import Path
import os
import re
import demoji
from datasets import Dataset, Features, Value, ClassLabel, DatasetDict
import dotenv

CURRENT_PATH = Path(__file__).parent

dotenv.load_dotenv(dotenv.find_dotenv())

df = pd.read_csv(os.path.join(CURRENT_PATH, "data", "wpp_2020.csv"))
df['label'] = 'pt-BR'

#Iterate over the rows of the dataframe
for index, row in df.iterrows():
    text = row['text'].strip()
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = demoji.replace(text, '')

    df.at[index, 'text'] = text

df = df[['text', 'label']]

dataset = Dataset.from_pandas(df, split='train', features=Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["pt-PT", "pt-BR"])
}))

dataset.train_test_split(test_size=0.2, shuffle=True, seed=42).push_to_hub(
    'fake_news_wpp_br', token=os.getenv("HF_TOKEN"))
