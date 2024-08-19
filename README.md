# Portuguese Variety Identifier

A classifier for distinguishing between European and Brazilian Portuguese.

## Development Environment

### 1. Set Up a Virtual Environment

To create a virtual environment using Conda, run the following commands:

```sh
conda create --name .conda python=3.11
conda activate .conda
```

### 2. Install Dependencies

Install the necessary dependencies by running:

```sh
pip install -e .
```

## Training

Training scripts are located in the [train](scripts/train/) folder. For example, to find the best hyperparameters for the BERT model, run:

```sh
sh scripts/train/search_bert.sh
```

## Evaluation

To evaluate the trained models, execute the evaluation script:

```sh
sh scripts/eval/run.sh
```
