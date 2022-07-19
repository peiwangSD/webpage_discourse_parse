# -*- coding: utf-8 -*-
import torch
from torch import nn
from module.paragraph_encoder import Paragraph_Encoder
from module.self_pointer_network import Sequence_Modeling_Network, Pairwise_Pointer_Network
from module.pair_classifier import Pair_Classifier, father_id_to_previous_id


class BaselineModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.paragraph_encoder = Paragraph_Encoder(args, config)
        if args.additional_encoder:
            print("use contextual encoder")
            self.sequence_modeling_network = Sequence_Modeling_Network(d_model=self.paragraph_encoder.hidden_dim, model_type=args.additional_encoder_type)
    
        self.father_net = Pairwise_Pointer_Network(input_dims=self.paragraph_encoder.hidden_dim, loss_type=args.loss_type, layer_type=args.layer_type)
        self.previous_net = Pair_Classifier(input_dims=2 * self.paragraph_encoder.hidden_dim)

        # self.dummy_node_embedding = nn.Embedding(1, self.paragraph_encoder.hidden_dim)
        self.dummy_node_embedding = torch.randn(self.paragraph_encoder.hidden_dim, requires_grad=True).to(args.device)  # TODO: the tensor you manually created by torch's tensor creating method is on cpu by default

        # self.alpha = alpha
        self.alpha = args.alpha
        self.additional_encoder = args.additional_encoder  # Whether to use a Sequence_Modeling_Network layer

    def forward(self, input_ids, input_mask=None, padding=None, golden_parent=None, golden_previous=None):
        encodings = self.paragraph_encoder(input_ids, input_mask)
        if self.additional_encoder:
            encodings = self.sequence_modeling_network(encodings,)
        encodings = torch.cat((self.dummy_node_embedding.unsqueeze(0).unsqueeze(0), encodings), dim=1)
        (father_node_logit_scores, father_loss) = self.father_net(encodings, padding, golden_parent)

        if golden_parent is None:  # inference
            fathers = torch.argmax(father_node_logit_scores, dim=2)
            fathers = fathers.tolist()
            # fathers = fathers - 1  # wipe off dummy node
        else:  # train, teacher forcing
            fathers = golden_parent + 1
        # print(fathers)
        previous_id_list = father_id_to_previous_id(fathers)
        # print(previous_id_list)
        previous_id_list = torch.tensor(previous_id_list).to(input_ids.device)
        (previous_relations_logit_scores, previous_loss) = self.previous_net(encodings, previous_id_list, golden_previous)

                    
        # fathers = fathers - 1  # wipe off dummy node
        if golden_parent is not None:
            if previous_loss is None:
                previous_loss = torch.tensor(0).to(input_ids.device)  # TODO 0704
            loss = self.alpha * father_loss + (1 - self.alpha) * previous_loss
            return (loss, father_loss, previous_loss, torch.argmax(father_node_logit_scores, dim=2) - 1, previous_relations_logit_scores)
        else:
            return (None, None, None, torch.argmax(father_node_logit_scores, dim=2) - 1, previous_relations_logit_scores)
