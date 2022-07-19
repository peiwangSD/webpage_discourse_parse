# -*- coding: utf-8 -*-
import torch
from torch import nn
from transformers import PreTrainedModel

from transformers import BertModel, BertConfig

from transformers import AutoModel
# from torch.cuda.amp import autocast


class Paragraph_Encoder(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        self.hidden_dim = config.hidden_size  # to get the hidden dim of PTLM

    # @autocast()
    def forward(self, input_ids, attention_mask=None,):
        """

        :param input_ids: [Batch_size, num_of_paragraph, max_seq_length]
        :param attention_mask:
        :return:
        """
        # one instance should become one batch
        batch_size, num_of_paragraph, max_seq_length = input_ids.shape
        input_ids = torch.reshape(input_ids, (-1, max_seq_length))
        attention_mask = torch.reshape(attention_mask, (-1, max_seq_length))
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        # pooled_output = outputs[0]
        last_hidden_state, pooler_output = outputs[:2]  # (batch_size, hidden_size)
        pooler_output = torch.reshape(pooler_output, (batch_size, num_of_paragraph, -1))

        return pooler_output
