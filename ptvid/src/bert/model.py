import torch
from transformers import BertModel


class LanguageIdentfier(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)

        self.linear = torch.nn.Linear(self.model.config.hidden_size, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        logits = self.sigmoid(logits)

        return logits
