from pathlib import Path
import os
import pandas as pd
import plotly.express as px


CURRENT_PATH = Path(__file__).parent


def load_autoencoder_bert_isolated():
    in_folder = os.path.join(
        CURRENT_PATH, 'raw', 'autoencoder_two_models_bert')
    out_folder = os.path.join(CURRENT_PATH, 'parsed')

    df_result = pd.DataFrame(
        columns=['domain', 'precision', 'recall', 'f1', 'accuracy'])

    for domain in ['law', 'literature', 'news', 'politics', 'social_media', 'web']:
        df_result = pd.concat([df_result, pd.read_json(os.path.join(
            in_folder, f"{domain}_train.json"))], ignore_index=True)

    df_result.to_json(os.path.join(out_folder, 'all_results',
                      'autoencoder_two_models_bert.json'), orient='records', force_ascii=False, indent=4)

    best_f1_rows = df_result.loc[df_result.groupby('domain')['f1'].idxmax(), [
        'domain', 'accuracy', 'f1', 'precision', 'recall']]

    best_f1_rows.to_json(os.path.join(out_folder, 'best_results',
                         'autoencoder_two_models_bert.json'), orient='records', force_ascii=False, indent=4)

    return df_result, best_f1_rows


def load_autoencoder_bert_ensemble():
    in_folder = os.path.join(
        CURRENT_PATH, 'raw', 'autoencoder_two_models_bert_ensemble')
    out_folder = os.path.join(CURRENT_PATH, 'parsed')

    df = pd.read_json(os.path.join(
        in_folder, f"ensemble_bert_autoencoder.json"))

    df.to_json(os.path.join(out_folder, 'all_results', 'autoencoder_two_models_bert_ensemble.json'),
               orient='records', force_ascii=False, indent=4)

    df.to_json(os.path.join(out_folder, 'best_results', 'autoencoder_two_models_bert_ensemble.json'),
               orient='records', force_ascii=False, indent=4)


def bert_isolated():
    in_folder = os.path.join(CURRENT_PATH, 'raw', 'bert_isolated')

    out_folder = os.path.join(CURRENT_PATH, 'parsed')

    df_result = pd.DataFrame(
        columns=['domain', 'precision', 'recall', 'f1', 'accuracy', 'loss', 'epoch'])

    for domain in ['law', 'literature', 'news', 'politics', 'social_media', 'web']:
        df = pd.read_json(os.path.join(in_folder, f"results_{domain}.json"))

        columns = {
            'validation_loss': 'loss',
            'validation_f1': 'f1',
            'validation_accuracy': 'accuracy',
            'validation_precision': 'precision',
            'validation_recall': 'recall',
        }

        df.rename(columns=columns, inplace=True)

        df['domain'] = domain

        df_result = pd.concat([df_result, df], ignore_index=True)

    best_f1_rows = df_result.loc[df_result.groupby('domain')['f1'].idxmax(
    ), ['domain', 'accuracy', 'f1', 'precision', 'recall']]

    df_result.to_json(os.path.join(out_folder, 'all_results',
                      'bert_isolated.json'), orient='records', force_ascii=False, indent=4)

    best_f1_rows.to_json(os.path.join(out_folder, 'best_results',
                         'bert_isolated.json'), orient='records', force_ascii=False, indent=4)

    return df_result, best_f1_rows


def bert_ensemble():
    in_folder = os.path.join(CURRENT_PATH, 'raw', 'bert_ensemble')
    out_folder = os.path.join(CURRENT_PATH, 'parsed')

    df = pd.read_json(os.path.join(
        in_folder, f"results_ensembler_embeddings_final.json"))

    df.to_json(os.path.join(out_folder, 'all_results', 'bert_ensemble.json'),
               orient='records', force_ascii=False, indent=4)

    # TODO: Deal With Reduction -> I know Mean is the best. But I want to Show it
    df = df[df['reduction'] == 'mean']

    best_f1_rows = df.loc[df.groupby(['domain'])['f1'].idxmax(
    ), ['domain', 'accuracy', 'f1', 'precision', 'recall', 'reduction']]

    best_f1_rows.to_json(os.path.join(out_folder, 'best_results',
                         'bert_ensemble.json'), orient='records', force_ascii=False, indent=4)

    return df, best_f1_rows


def n_gram_isolated():
    in_folder = os.path.join(CURRENT_PATH, 'raw', 'n_grams_isolated')

    df = pd.read_json(os.path.join(
        in_folder, f"output_metrics.json"))

    df_filtered = pd.DataFrame(
        columns=['domain', 'analyzer', 'f1', 'precision', 'recall', 'accuracy'])

    for _, row in df.iterrows():
        correct_result = None

        for result in row['results']:
            if result['test_domain'] == row['train_domain']:
                correct_result = result
                break

        df_filtered = pd.concat([df_filtered, pd.DataFrame({
            'domain': row['train_domain'],
            'analyzer': row['analyzer'],
            'f1': correct_result['f1'],
            'precision': correct_result['precision'],
            'recall': correct_result['recall'],
            'accuracy': correct_result['accuracy']
        }, index=[0])], ignore_index=True)

    best_f1_rows = df_filtered.loc[df_filtered.groupby('domain')['f1'].idxmax(
    ), ['domain', 'accuracy', 'f1', 'precision', 'recall', 'analyzer']]

    df_filtered.to_json(os.path.join(CURRENT_PATH, 'parsed', 'all_results',
                        'n_gram_isolated.json'), orient='records', force_ascii=False, indent=4)

    best_f1_rows.to_json(os.path.join(CURRENT_PATH, 'parsed', 'best_results',
                         'n_gram_isolated.json'), orient='records', force_ascii=False, indent=4)


def n_gram_ensemble():
    in_folder = os.path.join(CURRENT_PATH, 'raw', 'n_grams_ensemble')

    df = pd.read_json(os.path.join(
        in_folder, f"results_ensembler_n_grams.json"))

    df_filtered = pd.DataFrame(
        columns=['domain', 'strategy', 'f1', 'precision', 'recall', 'accuracy'])

    df = df.transpose()

    df.drop(columns=['max'], inplace=True)

    for _, row in df.iterrows():
        for key in df.columns:
            df_filtered = pd.concat([df_filtered, pd.DataFrame({
                'domain': row.name,
                'strategy': key,
                'f1': row[key]['f1'],
                'precision': row[key]['precision'],
                'recall': row[key]['recall'],
                'accuracy': row[key]['accuracy']
            }, index=[0])], ignore_index=True)

    df_filtered.to_json(os.path.join(CURRENT_PATH, 'parsed', 'all_results',
                                     'n_gram_ensemble.json'), orient='records', force_ascii=False, indent=4)

    df_filtered = df_filtered[df_filtered['strategy'] == 'majority_voting']

    df_filtered.to_json(os.path.join(CURRENT_PATH, 'parsed', 'best_results',
                        'n_gram_ensemble.json'), orient='records', force_ascii=False, indent=4)


def produce_plots_isolated():
    best_results_folder = os.path.join(CURRENT_PATH, 'parsed', 'best_results')

    for filename in ['autoencoder_two_models_bert', 'bert_isolated', 'n_gram_isolated']:
        df = pd.read_json(os.path.join(
            best_results_folder, f'{filename}.json'))

        df = df[df['train_domain'] != 'dslcc']

        f1_final = []

        labels = ['law', 'literature', 'news',
                  'politics', 'social_media', 'web']

        for idx1, train_domain in enumerate(labels):
            accumulator = []
            mean_reduction = 0

            df_train_domain = df[df['train_domain'] == train_domain]

            for idx2, test_domain in enumerate(labels):
                accumulator.append(
                    df_train_domain[df_train_domain['test_domain']
                                    == test_domain]['f1'].values[0]
                )

            # Calculate the Mean Reduction for the Train Domain
            specialist_f1 = accumulator[idx1]

            for idx3, f1 in enumerate(accumulator):
                if idx3 != idx1:
                    mean_reduction += (specialist_f1 - f1)

            mean_reduction /= len(labels) - 1

            accumulator.append(mean_reduction)

            f1_final.append(accumulator)

        if filename == 'autoencoder_two_models_bert':
            title = "AutoEncoder Isolated"
        elif filename == 'bert_isolated':
            title = "BERT Isolated"
        elif filename == 'n_gram_isolated':
            title = "N-Gram Isolated"

        fig = px.imshow(f1_final, text_auto='.2f', aspect="auto", labels=dict(
            x="Train Domain", y="Test Domain", color="F1-Score"), x=labels + ['mean_reduction'], y=labels, color_continuous_scale="Blues", range_color=[0, 1])

        fig.update_layout(xaxis_title="Test Domain", yaxis_title="Train Domain", autosize=True,
                          title=title, font=dict(size=16, family='Helvetica'), margin=dict(l=100, r=25, t=50, b=80))

        # Title Size
        fig.update_layout(title_font_size=24)

        fig.update_yaxes(automargin=False)

        fig.update_xaxes(automargin=False)

        fig.write_html(os.path.join(
            CURRENT_PATH, 'out', f'{filename}_f1.html'))

        fig.write_image(os.path.join(
            CURRENT_PATH, 'out', f'{filename}_f1.pdf'))
        """

"""


def produce_plots_ensemble():
    pass


def produce_plots_all_mixed():
    df = pd.read_json(os.path.join(CURRENT_PATH, 'raw',
                      'all_mixed', 'all_mixed_results.json'))

    fig = px.bar(df, x="domain", y="f1", title='All-Mixed BERT', text_auto='.2f',
                 color_discrete_sequence=['steelBlue'] * 6, text='f1')

    fig.update_layout(xaxis_title="Domain", yaxis_title="F1-Score",
                      font=dict(size=16, family='Helvetica'))

    fig.update_xaxes(categoryorder='category ascending')
    fig.update_yaxes(range=[0, 1], dtick=0.1)

    fig.update_traces(textposition="outside", cliponaxis=False)

    fig.write_html(os.path.join(CURRENT_PATH, 'out', f'all_mixed_f1.html'))
    fig.write_image(os.path.join(CURRENT_PATH, 'out', f'all_mixed_f1.pdf'))


def load_data():
    pass
    #load_autoencoder_bert_ensemble()

    #load_autoencoder_bert_isolated()

    #bert_isolated()

    #bert_ensemble()

    #n_gram_isolated()

    #n_gram_ensemble()


def main():
    #load_data()
    #produce_plots_isolated()
    #produce_plots_ensemble()
    produce_plots_all_mixed()


if __name__ == "__main__":
    main()
