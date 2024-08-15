import json
import re

import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import torch

from ptvid.constants import ROOT
from ptvid.src.bert.model import LanguageIdentifier


BERT_RESULTS_DIR = ROOT.parent / "results" / "bert"
BERT_MODELS_DIR = ROOT.parent / "model" / "bert"

model_keyname = "all_pos_prob=0.0_ner_prob=0.0"
modelpath = BERT_MODELS_DIR / model_keyname / "model.pth"
model = LanguageIdentifier()
checkpoint = torch.load(modelpath)
model.load_state_dict(checkpoint)
model.predict(["Isto é um teste.", "O cara apanhou o autocarro"])
model.predict(["O tipo apanhou o autocarro"])