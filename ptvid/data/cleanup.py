from datasets import load_dataset, Dataset, load_from_disk
from cleantext import clean
from ftlangdetect import detect
from tqdm import tqdm


class Cleanup:
    def __init__(self, dataset_raw_name, dataset_cleaned_name, DOMAINS) -> None:
        self.dataset_raw_name = dataset_raw_name
        self.dataset_cleaned_name = dataset_cleaned_name
        self.DOMAINS = DOMAINS
        self.FAST_TEXT_THRESHOLD = 0.7
        self.datasets = {}

    def clean_nan(self, df_dataset):
        print(f"Number of rows before dropping NaN: {len(df_dataset)}")

        df_dataset = df_dataset.dropna()

        print(f"Number of rows after dropping NaN: {len(df_dataset)}")

        return df_dataset

    def clean_non_portuguese(self, text):
        detected_lang = detect(text)

        if detected_lang["lang"] != "pt" or detected_lang["score"] < self.FAST_TEXT_THRESHOLD:
            return False

        return True

    def clean_empty_strings(self, df_dataset):
        print(f"Number of rows before dropping empty strings: {len(df_dataset)}")

        df_dataset = df_dataset[df_dataset["text"] != ""]

        print(f"Number of rows after dropping empty strings: {len(df_dataset)}")

        return df_dataset

    def clean_duplicates(self, df_dataset):
        print(f"Number of rows before dropping duplicates: {len(df_dataset)}")

        df_dataset = df_dataset.drop_duplicates(
            subset=["text"],
            keep="first",
        )

        print(f"Number of rows after dropping duplicates: {len(df_dataset)}")

        return df_dataset

    def clean_outliers(self, text, domain):
        thresholds = {
            "politics": (100, 3_000),
            "journalistic": (100, 1_500),
            "legal": (100, 1_500),
            "literature": (100, 1_500),
            "social_media": (10, 1_000),
            "web": (100, 5_000),
        }

        return thresholds[domain][0] < len(text) < thresholds[domain][1]

    def clean_text(self, text):
        return clean(
            text,
            fix_unicode=True,
            to_ascii=False,
            lower=False,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=True,
            no_digits=True,
            no_currency_symbols=True,
            no_punct=False,
            strip_lines=True,
            normalize_whitespace=True,
            no_emoji=True,
            lang="en",
        )

    def run(self, domain):
        print(f"Processing domain: {domain}...")

        df_dataset = load_dataset(self.dataset_raw_name, domain, split="train").to_pandas()

        tqdm.pandas()

        print(f"{domain} |Cleaning outliers...")
        outliers_indices = df_dataset["text"].progress_apply(lambda x: self.clean_outliers(x, domain))
        df_dataset = df_dataset[outliers_indices]
        hf_dataset = Dataset.from_pandas(df_dataset).select_columns(["text", "label"])

        print(f"Number of rows after cleaning: {len(hf_dataset)}")
        hf_dataset.save_to_disk(f"{self.dataset_cleaned_name}/{domain}")

    def push_to_hub(self):
        for domain in self.DOMAINS:
            hf_dataset = load_from_disk(f"{self.dataset_cleaned_name}/{domain}")

            hf_dataset.push_to_hub(
                self.dataset_cleaned_name,
                config_name=domain,
            )


if __name__ == "__main__":
    domains = ["politics", "web", "journalistic", "legal", "literature", "social_media"]

    clean_up = Cleanup(
        dataset_raw_name="arubenruben/portuguese-language-identification-cleaned-1",
        dataset_cleaned_name="arubenruben/portuguese-language-identification-cleaned",
        DOMAINS=domains,
    )

    for domain in domains:
        clean_up.run(domain)

    clean_up.push_to_hub()
