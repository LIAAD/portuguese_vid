from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from nltk.tokenize import sent_tokenize


def social_media():
    portuguese_elections = load_dataset(
        "arubenruben/portuguese_presidential_elections_tweets_lid"
    )

    fake_news_whatsapp = load_dataset(
        "arubenruben/fake_news_wpp_br"
    )

    hate_br = load_dataset(
        "arubenruben/hate_br_li"
    )

    final_dataset = DatasetDict(
        {
            'train': concatenate_datasets(
                [
                    portuguese_elections['train'],
                    fake_news_whatsapp['train'],
                    hate_br['train'],
                    portuguese_elections['test'],
                    fake_news_whatsapp['test'],
                    hate_br['test']
                ]
            )
        }
    )

    final_dataset = final_dataset.shuffle()

    final_dataset.push_to_hub(
        'arubenruben/portuguese-language-identification-raw',
        config_name='social_media',
    )


def journalistic():
    cetem = load_dataset(
        "arubenruben/cetem"
    )
    cetem = DatasetDict({
        'train': concatenate_datasets(
            [cetem['train'], cetem['test']]
        )
    })

    cetem = cetem.shuffle()

    cetem.push_to_hub(
        'arubenruben/portuguese-language-identification-raw',
        config_name='journalistic',
    )


def politics():
    europarl_pt = load_dataset("arubenruben/europarl", "pt", split="train")
    europarl_pt = europarl_pt.to_pandas()
    # Add label column
    europarl_pt['label'] = [0] * europarl_pt.shape[0]

    # Remove first and last 2 sentences
    europarl_pt['text'] = europarl_pt.apply(
        lambda x: ' '.join(sent_tokenize(x['text'])[2:-2]), axis=1)

    europarl_pt = Dataset.from_pandas(europarl_pt, split="train")
    europarl_pt = europarl_pt.select_columns(['text', 'label'])

    brazilian_senate = load_dataset("arubenruben/brazilian_senate_speeches")

    brazilian_senate = concatenate_datasets(
        [brazilian_senate['train'], brazilian_senate['test']])

    brazilian_senate = brazilian_senate.to_pandas()

    # Remove first and last 5 sentences
    brazilian_senate['text'] = brazilian_senate.apply(
        lambda x: ' '.join(sent_tokenize(x['text'])[5:-5]), axis=1)

    brazilian_senate = Dataset.from_pandas(brazilian_senate)
    brazilian_senate = brazilian_senate.select_columns(['text', 'label'])

    europarl_pt = europarl_pt.cast(brazilian_senate.features)

    politics = concatenate_datasets(
        [europarl_pt, brazilian_senate])

    politics = politics.select_columns(['text', 'label'])

    politics = politics.shuffle()

    politics.push_to_hub(
        'arubenruben/portuguese-language-identification-raw',
        config_name='politics',
    )


def legal():
    lener_br = load_dataset("lener_br")
    lener_br = concatenate_datasets(
        [lener_br['train'], lener_br['test'], lener_br['validation']])

    lener_br = lener_br.to_pandas()

    # Tokens column is a list of tokens, so we need to join them
    lener_br['text'] = lener_br['tokens'].apply(
        lambda tokens: ' '.join(tokens))

    lener_br['label'] = [1] * lener_br.shape[0]

    lener_br = Dataset.from_pandas(lener_br)

    lener_br = lener_br.select_columns(['text', 'label'])

    portuguese_legal_sentences = load_dataset(
        "stjiris/portuguese-legal-sentences-v0"
    )

    portuguese_legal_sentences = portuguese_legal_sentences.rename_column(
        original_column_name='sentence',
        new_column_name='text'
    )

    portuguese_legal_sentences = concatenate_datasets(
        [portuguese_legal_sentences['train'], portuguese_legal_sentences['test'], portuguese_legal_sentences['validation']])

    portuguese_legal_sentences = portuguese_legal_sentences.to_pandas()

    portuguese_legal_sentences['label'] = [0] * \
        portuguese_legal_sentences.shape[0]

    portuguese_legal_sentences = Dataset.from_pandas(
        portuguese_legal_sentences)

    portuguese_legal_sentences = portuguese_legal_sentences.select_columns([
                                                                           'text', 'label'])

    hf_dataset = DatasetDict({
        'train': concatenate_datasets(
            [lener_br, portuguese_legal_sentences]
        )
    })

    hf_dataset = hf_dataset.shuffle()

    hf_dataset.push_to_hub(
        'arubenruben/portuguese-language-identification-raw',
        config_name='legal',
    )


# social_media()
politics()
# journalistic()
# legal()
