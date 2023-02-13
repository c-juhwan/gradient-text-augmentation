# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
# Huggingface Modules
from transformers import AutoConfig, AutoModel
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name

class ClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(ClassificationModel, self).__init__()
        self.args = args

        # Define model
        huggingface_model_name = get_huggingface_model_name(self.args.model_type)
        self.config = AutoConfig.from_pretrained(huggingface_model_name)
        if args.model_ispretrained:
            self.model = AutoModel.from_pretrained(huggingface_model_name)
        else:
            self.model = AutoModel.from_config(self.config)

        self.embed_size = self.model.config.hidden_size
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = self.args.num_classes

        # Define classifier - As we need to train the model using soft labels, we need to define a custom classifier rather than using BERTforSequenceClassification
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.args.dropout_rate),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        cls_output = model_output.last_hidden_state[:, 0, :] # (batch_size, hidden_size)
        classification_logits = self.classifier(cls_output) # (batch_size, num_classes)

        return classification_logits
