import datasets
import pytest

from ptvid.src.data import Data
from ptvid.constants import DATASET_NAME


@pytest.fixture
def data():
    return Data(dataset_name=DATASET_NAME, split="train")


def test_load_domain_all(data):
    domain = "all"
    balance = False
    delexicalize = True
    pos_prob = 0.5
    ner_prob = 0.7
    sample_size = 100

    dataset = data.load_domain(domain, balance, delexicalize, pos_prob, ner_prob, sample_size)

    assert len(dataset) == sample_size
    assert isinstance(dataset, datasets.Dataset)


def test_load_domain_single(data):
    domain = "domain1"
    balance = True
    delexicalize = False
    pos_prob = None
    ner_prob = None
    sample_size = None

    dataset = data.load_domain(domain, balance, delexicalize, pos_prob, ner_prob, sample_size)

    assert len(dataset) > 0
    assert isinstance(dataset, datasets.Dataset)


def test_load_test_set(data):
    test_set = data.load_test_set()

    assert isinstance(test_set, dict)
    assert len(test_set) == len(data.DOMAINS)
    for domain in data.DOMAINS:
        assert domain in test_set
        assert isinstance(test_set[domain], datasets.Dataset)
