from data import load_train_data
from huggingface_hub import login
import dotenv
import os
from pathlib import Path
from trainer import Trainer

CURRENT_PATH = Path(__file__).parent

dotenv.load_dotenv(dotenv.find_dotenv())

login(token=os.getenv("HF_TOKEN"))


def main():

    n_epochs = 20
    lr = 1e-5

    for domain in ["dslcc"]:
        
        print(f'Training {domain} domain')

        brazilian_train, european_train = load_train_data(domain=domain)

        trainer = Trainer(european_train=european_train, brazilian_train=brazilian_train, lr=lr, n_epochs=n_epochs, domain=domain)

        df_results_train = trainer.train()

        df_results_train['domain'] = domain
        df_results_train['lr'] = lr

        df_results_train.to_json(os.path.join(
            CURRENT_PATH, 'out', f'{domain}_train.json'), orient='records')
                


if __name__ == '__main__':
    main()
