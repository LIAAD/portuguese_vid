import torch
from pathlib import Path
import os
from transformers import BertModel, BertForTokenClassification

import numpy as np

CURRENT_PATH = Path(__file__).parent


def load_models():
    models = []

    for model in ['law', 'literature', 'news', 'politics', 'social_media', 'web']:
        model = torch.load(os.path.join(CURRENT_PATH, 'models', f'{model}.pt'))
        models.append(model)

    return models


class LanguageIdentifer(torch.nn.Module):
    def __init__(self, mode='horizontal_stacking', pos_layers_to_freeze=0, bertimbau_layers_to_freeze=0):
        super().__init__()

        self.labels = ['pt-PT', 'pt-BR']

        self.portuguese_model = BertModel.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")

        self.portuguese_pos_tagging_model = BertForTokenClassification.from_pretrained(
            "lisaterumi/postagger-portuguese")

        for layer in range(bertimbau_layers_to_freeze):
            for name, param in self.portuguese_model.named_parameters():
                if f".{layer}" in name:
                    print(f"Freezing Layer {name} of Bertimbau")
                    param.requires_grad = False

        for layer in range(pos_layers_to_freeze):
            for name, param in self.portuguese_pos_tagging_model.named_parameters():
                if f".{layer}" in name:
                    print(f"Freezing Layer {name} of POS")
                    param.requires_grad = False

        self.portuguese_pos_tagging_model.classifier = torch.nn.Identity()
        self.mode = mode

        if self.mode == 'horizontal_stacking':
            self.linear = self.common_network(torch.nn.Linear(
                self.portuguese_pos_tagging_model.config.hidden_size + self.portuguese_model.config.hidden_size, 512))
        elif self.mode == 'bertimbau_only' or self.mode == 'pos_only' or self.mode == 'vertical_sum':
            self.linear = self.common_network(torch.nn.Linear(
                self.portuguese_model.config.hidden_size, 512))
        else:
            raise NotImplementedError

    def common_network(self, custom_linear):
        return torch.nn.Sequential(        
            custom_linear,
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 1),
        )

    def forward(self, input_ids, attention_mask):

        #(Batch_Size,Sequence Length, Hidden_Size)
        outputs_bert = self.portuguese_model(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

        #(Batch_Size,Sequence Length, Hidden_Size)
        outputs_pos = self.portuguese_pos_tagging_model(
            input_ids=input_ids, attention_mask=attention_mask).logits[:, 0, :]

        if self.mode == 'horizontal_stacking':
            outputs = torch.cat((outputs_bert, outputs_pos), dim=1)
        elif self.mode == 'bertimbau_only':
            outputs = outputs_bert
        elif self.mode == 'pos_only':
            outputs = outputs_pos
        elif self.mode == 'vertical_sum':
            outputs = outputs_bert + outputs_pos
            outputs = torch.nn.functional.normalize(outputs, p=2, dim=1)
        
        return self.linear(outputs)

class EnsembleModel(torch.nn.Module):
    def __init__(self, domain=['law', 'literature', 'news', 'politics', 'social_media', 'web']):
        super().__init__()
        
        self.domain = domain

        self.models = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'Ensemble Model running on {self.device}')

        for domain in self.domain:
            model_state_dict = torch.load(os.path.join(
                CURRENT_PATH, 'models', f'{domain}.pt'), map_location=self.device)

            new_model = LanguageIdentifer(mode='pos_only')

            new_model.load_state_dict(model_state_dict)

            new_model = new_model.to(self.device)

            self.models.append({
                'model': new_model,
                'domain': domain,
            })

    def forward(self, input_ids, attention_mask):
        results = []

        for model in self.models:
        
            model = model['model']
            
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            
            results.append(output)                

        return torch.stack(results, dim=1)

