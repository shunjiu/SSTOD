import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelWithLMHead, AutoModel

from models.TFIDF import TfIdf


class UBAR_plus(nn.Module):
    def __init__(self, args, tokenizer, device):
        super(UBAR_plus, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.device = device
        self.kp_model = TfIdf()
        self.gpt_model = AutoModelWithLMHead.from_pretrained(args.model_path)
        self.gpt_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, inputs_ids):
        return self.gpt_model(inputs_ids)

    def generate(self, **kwargs):
        return self.gpt_model.generate(**kwargs)

    def generate_correct_char(self, input_ids):
        """query the database and select the most similarity chars"""
        KP_results = self.kp_model.search_batch(input_ids)
        if len(KP_results) > 10:
            return KP_results[:10]
        return KP_results

