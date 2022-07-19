# -*- coding: utf-8 -*-
import torch
from torch import nn, relu
# from margin_loss import MarginLoss


class Sequence_Modeling_Network(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, model_type="transformer"):
        """

        :param d_model: embedding dim
        :param nhead:
        :param num_layers:
        """
        super().__init__()
        self.model_type = model_type
        if model_type == "transformer":
            print("transformer encoder")
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        elif model_type == "gru":
            print("gru encoder")
            self.encoder = nn.GRU(input_size=d_model, hidden_size=d_model//2, num_layers=num_layers, batch_first=True, bidirectional=True)

        elif model_type == "lstm":
            print("lstm encoder")
            self.encoder = nn.LSTM(input_size=d_model, hidden_size=d_model//2, num_layers=num_layers, batch_first=True, bidirectional=True)
        
        else:
            print("FFNN after PTLM")
            self.encoder = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.Tanh()
            )
    
    def forward(self, inputs, mask=None):
        """

        :param inputs: (Batch_size, num_paragraph, hiddem_dim)
        :param mask:
        """
        # print(inputs.shape)
        if self.model_type == "transformer":
            return self.encoder(src=inputs, mask=mask)
        elif self.model_type == "gru":
            output, h_n = self.encoder(inputs)
            return output
        elif self.model_type == "lstm":
            output, (h_n, c_n) = self.encoder(inputs)
            return output
        else:
            return self.encoder(inputs)
                


class Pairwise_Pointer_Network(nn.Module):
    def __init__(self, input_dims=512, loss_type="margin", layer_type="bilinear", max_paragraph_num=200):
        super().__init__()
        self.input_dims = input_dims
        self.head_mlp = nn.Sequential(nn.Linear(input_dims, input_dims),
                                      nn.ReLU(),
                                      nn.Linear(input_dims, input_dims),
                                      nn.Tanh()
                                      )
        self.tail_mlp = nn.Sequential(nn.Linear(input_dims, input_dims),
                                      nn.ReLU(),
                                      nn.Linear(input_dims, input_dims),
                                      nn.Tanh()
                                      )
        self.relative_position_encoding = nn.Embedding(max_paragraph_num, max_paragraph_num)  # TODO: 0703
        
        if layer_type=="bilinear":
            # use a bilinear layer to model the interaction between paragraph, and the relative positional encoding will be added at tail vector
            print("use bilinear")
            self.bilinear = nn.Bilinear(input_dims, input_dims + max_paragraph_num, 1)
            self.linear = None
        
        elif layer_type=="linear":
            # use a linear layer to model the interaction between paragraph, and the relative positional encoding will be concatenated together with head vector and tail vector
            print("use linear")
            self.linear = nn.Linear(2*input_dims+max_paragraph_num, 1)
            self.bilinear = None
        
        elif layer_type=="mixed":
            # use a multihead bilinear layer to model the interaction between paragraph before a final linear projection layer, and the relative positional encoding will be added at tail vector
            print("use mixed")
            self.MEDIM = 8
            self.bilinear = nn.Bilinear(input_dims, input_dims + max_paragraph_num, self.MEDIM)
            self.activate = nn.Tanh()
            self.linear = nn.Linear(self.MEDIM, 1)
        
        else:  # Multihead Attention
            # use the nn.MultiheadAttention implementation to model the interaction between paragraph
            print("use attention")
            self.mha = nn.MultiheadAttention(embed_dim=input_dims, num_heads=1,)# batch_first=True)
        self.layer_type = layer_type
        
        if loss_type == "margin":
            self.loss = nn.MultiMarginLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.loss_type = loss_type

    def forward(self, paragraphs, padding=None, golden=None):
        """
        `remember you still need to take batch dimension into consideration even if batch size = 1`
        Doing `masked` self attention
        :param paragraphs: (batch_size, num_paragraphs, hidden_dim)
        :param padding: upper triangular matrix, (batch_size, num_paragraphs, num_paragraphs)
        :param golden(Optional): (batch_size, num_paragraphs, num_paragraphs)  ### (batch_size, num_paragraphs)
        :return: father_node_scores: (batch_size, num_paragraphs, num_paragraphs)
        :return: loss: scalar
        """
        if len(paragraphs.shape) == 2:
            paragraphs = paragraphs.unsqueeze(dim=0)
        assert len(paragraphs.shape) == 3

        loss = None
        # 矩阵运算，不要写for循环
        head_vectors = self.head_mlp(paragraphs)  # (Batch, num_paragraph, hidden_dim)
        tail_vectors = self.tail_mlp(paragraphs)  # (Batch, num_paragraph, hidden_dim)
        batch_size = paragraphs.shape[0]
        num_paragraphs = paragraphs.shape[1]

        # TODO: Refactor:
        if self.layer_type=="bilinear":
            # Deliminated:
            """father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(paragraphs.device)
            for t in range(num_paragraphs):
                # print(0)
                for h in range(t):
                    # print(1)
                    tails = tail_vectors[:,t,:]
                    heads = head_vectors[:,h,:]
                    # father_node_logit_matrix[:,h,t] = self.bilinear(heads, tails)
                    father_node_logit_matrix[:,t,h] = self.bilinear(heads, tails)"""
            
            # Parallel implementation:
            """# head_vectors = head_vectors.repeat(1, num_paragraphs, 1)  # (N, num_paragraphs*num_paragraphs, hidden_dims)
            head_vectors = head_vectors.unsqueeze(1).expand(-1, num_paragraphs, -1, -1).reshape(batch_size, num_paragraphs*num_paragraphs, self.input_dims)
            # tail_vectors = tail_vectors.unsqueeze(2).repeat(1, 1, num_paragraphs, 1).reshape(batch_size, num_paragraphs*num_paragraphs, self.input_dims)  # (N, num_paragraphs*num_paragraphs, hidden_dims)
            tail_vectors = tail_vectors.unsqueeze(2).expand(-1, -1, num_paragraphs, -1).reshape(batch_size, num_paragraphs*num_paragraphs, self.input_dims)
            father_node_logit_matrix = self.bilinear(head_vectors, tail_vectors).squeeze(-1).reshape(batch_size, num_paragraphs, num_paragraphs)  # (N, num_paragraphs*num_paragraphs, 1) -> (N, num_paragraphs, num_paragraphs)
            attn_mask = torch.triu(torch.ones(num_paragraphs, num_paragraphs)).to(paragraphs.device)
            father_node_logit_matrix = father_node_logit_matrix * (1 - attn_mask)"""
            # head_vectors = head_vectors.repeat(1, num_paragraphs, 1)  # (N, num_paragraphs*num_paragraphs, hidden_dims)
            # head_vectors = head_vectors.unsqueeze(1).expand(-1, num_paragraphs, -1, -1).reshape(batch_size, num_paragraphs*num_paragraphs, self.input_dims)
            # tail_vectors = tail_vectors.unsqueeze(2).repeat(1, 1, num_paragraphs, 1).reshape(batch_size, num_paragraphs*num_paragraphs, self.input_dims)  # (N, num_paragraphs*num_paragraphs, hidden_dims)
            # tail_vectors = tail_vectors.unsqueeze(2).expand(-1, -1, num_paragraphs, -1).reshape(batch_size, num_paragraphs*num_paragraphs, self.input_dims)
            # father_node_logit_matrix = self.bilinear(head_vectors, tail_vectors).squeeze(-1).reshape(batch_size, num_paragraphs, num_paragraphs)  # (N, num_paragraphs*num_paragraphs, 1) -> (N, num_paragraphs, num_paragraphs)
            father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(paragraphs.device)
            for t in range(num_paragraphs):
                tails = tail_vectors[:,t,:].repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph, hidden_dim)
                
                # TODO: 0703 inject relative positional information
                relative_positions = list(range(t, 0, -1))
                relative_positions = relative_positions + [0] * (num_paragraphs - len(relative_positions))  # TODO: 0 stands for invalid father node (current node or the node after current node)
                relative_positions = [relative_positions for n in range(batch_size)]  # batchify
                relative_positions = torch.tensor(relative_positions, dtype=torch.long).to(paragraphs.device)
                relative_position_encodings = self.relative_position_encoding(relative_positions)  # (num_paragraph, 200)
                tails = torch.cat((tails, relative_position_encodings), -1)

                father_node_logit_matrix[:,t,:] = self.bilinear(head_vectors, tails).permute(0,2,1)  # (Batch, num_paragraph, 1) -> (Batch, 1, num_paragraph)
            attn_mask = torch.triu(torch.ones(num_paragraphs, num_paragraphs)).to(paragraphs.device)
            father_node_logit_matrix = father_node_logit_matrix * (1 - attn_mask)
            
        # TODO: linear
        elif self.layer_type=="linear":
            # Deliminated:
            """father_node_logit_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs).to(paragraphs.device)
            for t in range(num_paragraphs):
                # print(0)
                for h in range(t):
                    # print(1)
                    tails = tail_vectors[:,t,:]
                    heads = head_vectors[:,h,:]
                    concate = torch.cat((heads, tails), -1)
                    # father_node_logit_matrix[:,h,t] = self.bilinear(heads, tails)
                    father_node_logit_matrix[:,t,h] = self.linear(concate)"""
            
            # Parallel implementation:
            head_vectors = head_vectors.repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph**2, hidden_dim)
            tail_vectors = tail_vectors.unsqueeze(2).repeat(1, 1, num_paragraphs, 1).reshape(batch_size, num_paragraphs*num_paragraphs, self.input_dims)  # (Batch, num_paragraph**2, hidden_dim)

            # TODO: 0703 inject relative positional information
            huge_relative_positions = []
            for t in range(num_paragraphs):
                relative_positions = list(range(t, 0, -1))
                relative_positions = relative_positions + [0] * (num_paragraphs - len(relative_positions))  # TODO: 0 stands for invalid father node (current node or the node after current node)
                huge_relative_positions.append(relative_positions)
            huge_relative_positions = [huge_relative_positions for n in range(batch_size)]  # batchify, (Batch, num_paragraph, num_paragraph)
            huge_relative_positions = torch.tensor(huge_relative_positions, dtype=torch.long).to(paragraphs.device)
            relative_position_encodings = self.relative_position_encoding(huge_relative_positions).reshape(batch_size, num_paragraphs*num_paragraphs, -1)  # (Batch, num_paragraph**2, 200)

            concate = torch.cat((head_vectors, tail_vectors, relative_position_encodings), -1).reshape(batch_size, num_paragraphs, num_paragraphs, -1)  # (N, num_paragraphs*num_paragraphs, 2*hidden_dims) -> (N, num_paragraphs, num_paragraphs, 2*hidden_dims)
            father_node_logit_matrix = self.linear(concate).squeeze(-1)  # (N, num_paragraphs, num_paragraphs, 1) -> (N, num_paragraphs, num_paragraphs)
            attn_mask = torch.triu(torch.ones(num_paragraphs, num_paragraphs)).to(paragraphs.device)
            father_node_logit_matrix = father_node_logit_matrix * (1 - attn_mask)
        

        elif self.layer_type=="mixed":
            medium_matrix = torch.zeros(batch_size, num_paragraphs, num_paragraphs, self.MEDIM).to(paragraphs.device)
            for t in range(num_paragraphs):
                tails = tail_vectors[:,t,:].repeat(1, num_paragraphs, 1)  # (Batch, num_paragraph, hidden_dim)
                medium_matrix[:,t,:,:] = self.bilinear(head_vectors, tails).unsqueeze(1)  # (Batch, num_paragraph, 128) -> (Batch, 1, num_paragraph, 128)
            # medium_matrix = father_node_logit_matrix * (1 - attn_mask)

            father_node_logit_matrix = self.linear(self.activate(medium_matrix)).squeeze(-1)  # (N, num_paragraphs, num_paragraphs, 128) -> (N, num_paragraphs, num_paragraphs)
            attn_mask = torch.triu(torch.ones(num_paragraphs, num_paragraphs)).to(paragraphs.device)
            father_node_logit_matrix = father_node_logit_matrix * (1 - attn_mask)
        
        else:  # 6/30
            attn_mask = torch.triu(torch.ones(num_paragraphs, num_paragraphs)).to(paragraphs.device) # For a byte mask, a non-zero value indicates that the corresponding position is not allowed to attend
            attn_output, attn_output_weights = self.mha(query=paragraphs.permute(1,0,2), key=paragraphs.permute(1,0,2), value=paragraphs.permute(1,0,2), attn_mask=attn_mask)  # average_attn_weights=True, attn_output_weights: (N,L,S), N=1, num_heads=1, L=S=num_paragraph
            father_node_logit_matrix = attn_output_weights * (1 - attn_mask)

        # TODO
        """# head_vectors =   # (Batch, num_paragraph, hidden_dim)
        # tail_vectors =   # (Batch, num_paragraph, hidden_dim)
        father_node_logit_matrix = self.bilinear(head_vectors, tail_vectors)  # (Batch, num_paragraph, 1)"""
        # for h in range(num_paragraphs):



        # logits = self.bilinear(head_vectors, tail_vectors)
        """inf_mask = (1 - padding) * (-1.0e10)
        print(father_node_logit_matrix)
        father_node_logit_matrix = inf_mask * father_node_logit_matrix
        print(father_node_logit_matrix)""" # Do not need this padding (any more because batch_size=1 and the padding is by nature a triangle padding)
        # print(father_node_logit_matrix.shape)
        # father_node_logit_scores = torch.softmax(father_node_logit_matrix, dim=2)[:, 1:, :]  # wipe off dummy node
        father_node_logit_scores = torch.softmax(father_node_logit_matrix[:, 1:, :], dim=2)  # wipe off dummy node, but keep the sum of probabilities = 1
        # print(father_node_logit_scores.shape)

        # nn.MarginRankingLoss
        # print(golden)
        if golden is not None:
            # Compute loss
            # TODO:
            flatten_golden = torch.flatten(golden)
            flatten_golden = flatten_golden + 1  # Deal with "NA" type
            flatten_father_node_logit_scores = torch.flatten(father_node_logit_scores, end_dim=1)
            flatten_father_node_logit_matrix = torch.flatten(father_node_logit_matrix[:, 1:, :], end_dim=1)  # TODO
            # print(flatten_father_node_logit_scores.shape)
            # print(flatten_golden)
            # print(flatten_golden.shape)
            # print(flatten_father_node_logit_matrix.shape)
            # print(flatten_father_node_logit_scores)
            # loss = self.loss(flatten_father_node_logit_scores, flatten_golden)
            if self.loss_type == "ce":
                loss = self.loss(flatten_father_node_logit_matrix, flatten_golden)
            else:
                loss = self.loss(flatten_father_node_logit_scores, flatten_golden)

        outputs = (father_node_logit_scores, loss)
        return outputs

