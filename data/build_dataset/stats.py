from pathlib import Path
import os
import pandas as pd
import plotly.express as px
import logging

CURRENT_PATH = Path(__file__).parent


def plot_original_stats(dataset, domain):
    df = dataset.to_pandas()

    #Value counts
    value_counts = df['label'].value_counts()

    european_count = value_counts[0]
    brazilian_count = value_counts[1]

    df_plot = pd.DataFrame({
        'label': ['European', 'Brazilian'],
        'count': [european_count, brazilian_count]
    })

    logging.info(f"Plotting original stats {domain} dataset")

    fig = px.bar(df_plot, x='label', y='count',
                 title=f'Original Data Distribution {domain} Dataset')
    fig.write_html(os.path.join(CURRENT_PATH, "out", domain,
                    f"original_distribution.html"))

    #Token Counts
    logging.info(f"Calculating token count {domain} dataset")

    df['token_count'] = df['text'].apply(lambda x: len(x.split()))

    fig = px.histogram(df, x='token_count',
                       title=f'Token count distribution {domain} dataset')
    fig.write_html(os.path.join(CURRENT_PATH, "out", domain,
                    f"original_token_counter.html"))

    #Char Counts
    logging.info(f"Calculating char count {domain} dataset")

    df['char_count'] = df['text'].apply(lambda x: len(x))
    fig = px.histogram(df, x='char_count',
                       title=f'Char count distribution {domain} dataset')
    fig.write_html(os.path.join(CURRENT_PATH, "out", domain,
                    f"original_char_counter.html"))


def plot_final_stats(dataset_dict, domain):
    df_plot_labels = pd.DataFrame(columns=['label', 'count', 'split'])

    for split in ['train', 'validation', 'test']:
        df = dataset_dict[split].to_pandas()

        #Value counts
        value_counts = df['label'].value_counts()        

        european_count = value_counts[0]
        brazilian_count = value_counts[1]

        df_plot_labels = pd.concat([df_plot_labels, pd.DataFrame({
            'label': 'European',
            'count': european_count,
            'split': split
        }, index=[0])], ignore_index=True)

        df_plot_labels = pd.concat([df_plot_labels, pd.DataFrame({
            'label': 'Brazilian',
            'count': brazilian_count,
            'split': split
        }, index=[0])], ignore_index=True)

        #Token Counts
        df['token_count'] = df['text'].apply(lambda x: len(x.split()))

        fig = px.histogram(
            df, x='token_count', title=f'Token count distribution {domain} dataset {split}')
        fig.write_html(os.path.join(CURRENT_PATH, "out",
                        domain, f"final_token_counter_{split}.html"))
        

        #Reduce The X Axis Range
        fig.update_xaxes(range=[0, max(df['token_count'])])

        logging.debug(
            f"Max token count {max(df['token_count'])} Domain {domain} Split {split}")

        #Char Counts
        df['char_count'] = df['text'].apply(lambda x: len(x))
        fig = px.histogram(
            df, x='char_count', title=f'Char count distribution {domain} dataset {split}')

        #Reduce The X Axis Range
        fig.update_xaxes(range=[0, max(df['char_count'])])

        if domain == 'social_media':
            # Print the Element with the max char count
            print(df[df['char_count'] == max(df['char_count'])]
                  ['text'].values[0])

        logging.debug(
            f"Max char count {max(df['char_count'])} Domain {domain} Split {split}")

        fig.write_html(os.path.join(CURRENT_PATH, "out",
                        domain, f"final_char_counter_{split}.html"))
        

    logging.info(f"Plotting final stats {domain} dataset")

    fig = px.bar(df_plot_labels, x='label', y='count', color='split', barmode='group',
                 title=f'Final Data Distribution {domain} Dataset')
    fig.write_html(os.path.join(CURRENT_PATH, "out", domain,
                    f"final_distribution.html"))    
    
if __name__ == "__main__":
    from datasets import load_dataset

    for domain in ['literature', 'web', 'politics', 'social_media', 'law', 'news']:
        dataset = load_dataset("arubenruben/Portuguese_Language_Identification", domain)        
        
        plot_final_stats(dataset, domain)