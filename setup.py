from setuptools import setup, find_packages

setup(
    name='pt_variety_identifier',
    version='0.0.1',
    description='Identify the variety of Portuguese used in a text',
    install_requires=[
        'pandas',
        'datasets',
        'zstandard',
        'clean-text[gpl]',
        'fasttext-langdetect',
        'numpy',
        'tqdm',
        'imbalanced-learn',
        'spacy[cuda11x]',
        'evaluate',
        'nltk',
        'transformers',
        'torch'
    ],
    packages=find_packages(),
    author='Rúben Almeida'
)
