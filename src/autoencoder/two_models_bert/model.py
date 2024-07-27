import torch
from transformers import BertModel


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.bert = BertModel.from_pretrained(
            'neuralmind/bert-base-portuguese-cased').to(self.device)

        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size,
                            self.bert.config.hidden_size // 5),
            torch.nn.ReLU(),
            torch.nn.Linear(self.bert.config.hidden_size // 5,
                            self.bert.config.hidden_size // 10),
            torch.nn.ReLU(),
            torch.nn.Linear(self.bert.config.hidden_size // 10,
                            self.bert.config.hidden_size // 30),
            torch.nn.ReLU(),
        ).to(self.device)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size // 30,
                            self.bert.config.hidden_size // 10),
            torch.nn.ReLU(),
            torch.nn.Linear(self.bert.config.hidden_size // 10,
                            self.bert.config.hidden_size // 5),
            torch.nn.ReLU(),
            torch.nn.Linear(self.bert.config.hidden_size //
                            5, self.bert.config.hidden_size),
            torch.nn.Sigmoid()
        ).to(self.device)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state[:, 0, :]

        encoded = self.encoder(bert_output)

        decoded = self.decoder(encoded)

        return bert_output, decoded
