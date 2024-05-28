import torch.nn as nn
from transformers import BertModel, RobertaModel

bert_model_path = 'E:/projects/bert-base-uncased'
codebert_model_path = 'E:/projects/codebert-base'
roberta_model_path = 'E:/projects/roberta-base'

class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        # self.bert = BertModel.from_pretrained(bert_model_path)
        self.codebert = BertModel.from_pretrained(codebert_model_path)
        # self.roberta = RobertaModel.from_pretrained(roberta_model_path)
        self.linear = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        # hidden_out = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=False)
        hidden_out = self.codebert(input_ids, attention_mask=attention_mask, output_hidden_states=False)
        # hidden_out = self.roberta(input_ids, attention_mask=attention_mask, output_hidden_states=False)
        pred = self.linear(hidden_out.pooler_output)
        return pred
