
from datasets import DatasetDict, Dataset
from stats import plot_original_stats, plot_final_stats
import logging
import nltk
import pandas as pd


def extract_n_documents(dataset, n_validation, n_test):
    
    df = dataset.to_pandas()

    df_brazilian = df[df['label'] == 1]
    df_european = df[df['label'] == 0]
    
    test_df_brazilian = df_brazilian[:n_test]
    test_df_european = df_european[:n_test]

    val_df_brazilian = df_brazilian[n_test: n_validation]
    val_df_european = df_european[n_test: n_validation]

    train_df_brazilian = df_brazilian[n_validation:]
    train_df_european = df_european[n_validation:]

    # Value counts
    logging.info(
        f"Value counts train _social_media dataset, {len(train_df_brazilian)} brazilian, {len(train_df_european)} european")

    dataset = DatasetDict({
        'train': Dataset.from_pandas(pd.concat([train_df_brazilian, train_df_european], ignore_index=True), features=dataset.features, split='train'),
        'validation': Dataset.from_pandas(pd.concat([val_df_brazilian, val_df_european], ignore_index=True), features=dataset.features, split='validation'),
        'test': Dataset.from_pandas(pd.concat([test_df_brazilian, test_df_european], ignore_index=True), features=dataset.features, split='test')
    })

    return dataset


def generic_procedure(dataset, domain, ploting=False):

    if ploting:
        plot_original_stats(dataset, domain)

    train_test = dataset.train_test_split(
        test_size=0.3, stratify_by_column='label', seed=42, shuffle=True)

    train_validation = train_test['train'].train_test_split(
        test_size=0.3, stratify_by_column='label', seed=42, shuffle=True)

    dataset = DatasetDict({
        'train': train_validation['train'],
        'validation': train_validation['test'],
        'test': train_test['test']
    })

    if ploting:
        plot_final_stats(dataset, domain)

    return dataset


def process_web(dataset, ploting=False):
    return generic_procedure(dataset, 'web', ploting=ploting)


def process_politics(dataset, ploting=False):
    df = dataset.to_pandas()

    for index, row in df.iterrows():
        if index % 10000 == 0 and index > 0:
            logging.info(f"Processing row {index} of {len(df)}")

        # Remove 20% of the Brazilian starting text
        if row['label'] == 1:
            df.at[index, 'text'] = row['text'][int(len(row['text']) * 0.2):]

    df.reset_index(drop=True, inplace=True)

    dataset = Dataset.from_pandas(df, split='train', features=dataset.features)

    return generic_procedure(dataset, 'politics', ploting=ploting)


def process_social_media(dataset, ploting=False):
    if ploting:
        plot_original_stats(dataset, 'social_media')
    
    dataset = extract_n_documents(dataset, 500, 300)

    if ploting:
        plot_final_stats(dataset, 'social_media')

    return dataset


def process_law(dataset, ploting=False):
    if ploting:
        plot_original_stats(dataset, 'law')
    
    dataset = extract_n_documents(dataset, 600, 500)

    if ploting:
        plot_final_stats(dataset, 'law')

    return dataset
   


def process_news(dataset, ploting=False):
    return generic_procedure(dataset, 'news', ploting=ploting)


def process_literature(dataset, ploting=False):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')

    df = dataset.to_pandas()
    df_final = pd.DataFrame(columns=['text', 'label'])

    for _, row in df.iterrows():
        sentences = sent_tokenizer.tokenize(row['text'])
        # Create a new row for each 3 sentences
        for i in range(0, len(sentences), 3):
            if i + 3 > len(sentences):
                break

            df_final = pd.concat([df_final, pd.DataFrame({
                'text': ' '.join(sentences[i:i + 3]),
                'label': row['label']
            }, index=[0])])

    df_final.reset_index(drop=True, inplace=True)

    dataset = Dataset.from_pandas(
        df_final, features=dataset.features, split='train')

    return generic_procedure(dataset, 'literature', ploting=ploting)


def process_domain(dataset, domain, ploting=False):

    if domain == 'web':
        return process_web(dataset, ploting)

    if domain == 'politics':
        return process_politics(dataset, ploting)

    if domain == 'social_media':
        return process_social_media(dataset, ploting)

    if domain == 'law':
        return process_law(dataset, ploting)

    if domain == 'news':
        return process_news(dataset, ploting)

    if domain == 'literature':
        return process_literature(dataset, ploting)

    raise ValueError(f"Domain {domain} not found")
