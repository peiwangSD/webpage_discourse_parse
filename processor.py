from tqdm import tqdm
from typing import Dict, List

def convert_figure(tokens):
    return "图片"

NODE_IDENTITY_dict = {0:"Title", 1:"Content", 2:"Site", 3:"Meta", 4:"Decorative", 5:None}
NODE_MODAL_dict = {0:"Title", 1:"Text", 2:"Figure", 3:"Figure&Title", 4:None}
FATHER_RELATION_dict = {0:"Title", 1:"Attribute", 2:"Summary", 3:"Elaborate", 4:"Caption", 5:None}
PREVIOUS_RELATION_dict = {0:"Continue", 1:"Break", 2:"Combine", 3:None}

def text_to_id(inputs: List, id_to_text_mapper: Dict):
    if type(inputs) != list:
        inputs = [inputs]
    text_to_id_mapper = {v:k for k, v in id_to_text_mapper.items()}
    return [text_to_id_mapper[text] if text in text_to_id_mapper.keys() else text_to_id_mapper[None] for text in inputs]


class WebDataProcessor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        # self.label_field = labelfield
        # self.LABEL_TO_ID = labelfield.label2id
        self.text2id_mappers = {"Node_Identity": NODE_IDENTITY_dict,
                                "Node_modal": NODE_MODAL_dict,
                                "Father_Relation": FATHER_RELATION_dict,
                                "Previous_Relation": PREVIOUS_RELATION_dict}

    def get_features_from_dataset(self, dataset):
        print("Processing to get input features ...")
        features = []
        for instance in tqdm(dataset):
            feature = self.instance_to_features(instance)
            features.append(feature)
        return features

    def instance_to_features(self, instance):
        paragraphs = instance["Content"]
        paragraph_ids = []
        for tokens in paragraphs:
            if tokens.startswith("https://") or tokens.startswith("http://"):
                tokens = convert_figure(tokens) if self.args.preprocess_figure_node else tokens  # TODO: 0703
            sents = self.tokenizer.tokenize(tokens)
            sents = sents[:self.max_seq_length]
            input_ids = self.tokenizer.convert_tokens_to_ids(sents)
            paragraph_ids.append(input_ids)
        # relation_id = self.LABEL_TO_ID[instance["relation_type"]]
        # node_identity_ids = instance["Node_Identity"]
        # node_modal_ids = instance["Node_modal"]
        # father_relation_ids = instance["Father_Relation"]
        # previous_relation_ids = instance["Previous_Relation"]
        instance["Father"] = [int(f) if f!="NA" else -1 for f in instance["Father"]]
        # print(instance["Father"])

        instance.update({
            "input_ids": paragraph_ids,
        })
        for key, mapper in self.text2id_mappers.items():
            instance.update(
                {key+"_ids": text_to_id(instance[key], mapper)}
            )
        
        # instance["Previous_Relation_ids"] = [0 if id==3 else id for id in instance["Previous_Relation_ids"]]  # 类别数一致

        assert len(instance["Father"]) == len(paragraph_ids), instance["Father"]
        assert len(instance["Previous_Relation_ids"]) == len(paragraph_ids), instance["Previous_Relation"]
        
        return instance
