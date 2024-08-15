import torch
from transformers import BertModel, BertTokenizer

from ptvid.constants import MODEL_NAME


class LanguageIdentifier(torch.nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self._model_name = model_name
        self.model = BertModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        probs = self.sigmoid(logits)
        return probs

    def predict(self, texts: list[str]):
        tokenizer = BertTokenizer.from_pretrained(self._model_name)
        probs = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", )
            inputs.pop("token_type_ids")
            
            prob = self.forward(**inputs)
            probs.append(prob.item())
        return probs
