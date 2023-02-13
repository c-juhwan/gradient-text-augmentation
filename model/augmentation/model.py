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

class MainModel(nn.Module):
    """
    Input -> Encoder -> Sum     -> WAE -> Latent -> Classifier -> Output
                     -> Decoder -> Reconstruction -> Output
    """
    def __init__(self, args: argparse.Namespace) -> None:
        super(MainModel, self).__init__()
        self.args = args

        # Define model
        huggingface_model_name = get_huggingface_model_name(self.args.model_type)
        self.config = AutoConfig.from_pretrained(huggingface_model_name)
        if args.model_ispretrained:
            self.bart = AutoModel.from_pretrained(huggingface_model_name)
        else:
            self.bart = AutoModel.from_config(self.config)

        self.vocab_size = self.bart.config.vocab_size
        self.embed_size = self.bart.config.hidden_size
        self.hidden_size = self.bart.config.hidden_size
        self.latent_size = self.args.latent_size
        self.dropout_rate = self.args.dropout_rate
        self.max_seq_len = self.args.max_seq_len
        self.pad_token_id = self.args.pad_token_id
        self.sampling_strategy = self.args.sampling_strategy
        self.sampling_temperature = self.args.sampling_temperature
        self.sampling_topk = self.args.sampling_topk
        self.sampling_topp = self.args.sampling_topp

        self.encoder = self.bart.encoder
        self.decoder = self.bart.decoder
        self.decoder_vocab = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, self.vocab_size)
        )
        self.latent_encoder = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.latent_decoder = nn.Linear(self.hidden_size // 2, self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2 , self.hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.args.num_classes)
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Encode input_ids and attention_mask to hidden states
        """
        encoder_outputs = self.encoder(input_ids, attention_mask)
        encoder_outputs = encoder_outputs['last_hidden_state'] # (batch_size, max_seq_len, hidden_size)

        return encoder_outputs

    def process_latent_module(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """
        Encode encoder_outputs to latent vector
        """
        #encoder_refined = torch.sum(encoder_outputs, dim=1) # (batch_size, hidden_size)
        encoder_refined = torch.max(encoder_outputs, dim=1)[0] # (batch_size, hidden_size)

        latent_encoded = self.latent_encoder(encoder_refined) # (batch_size, latent_size)
        latent_decoded = self.latent_decoder(latent_encoded) # (batch_size, hidden_size)

        return latent_encoded, latent_decoded

    def classify(self, latent_decoded: torch.Tensor) -> torch.Tensor:
        """
        Classification using latent vector
        """
        classification_logits = self.classifier(latent_decoded) # (batch_size, num_classes)

        return classification_logits

    def decode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, encoder_outputs: torch.Tensor, latent_decoded: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to output_ids
        """
        decoder_input_id = self.shift_tokens_right(input_ids, self.pad_token_id, self.bart.config.decoder_start_token_id)

        latent_decoded = latent_decoded.unsqueeze(1).repeat(1, self.max_seq_len, 1) # (batch_size, max_seq_len, hidden_size)
        hidden_states = self.args.aug_encoder_latent_ratio * encoder_outputs + (1 - self.args.aug_encoder_latent_ratio) * latent_decoded # (batch_size, max_seq_len, hidden_size)

        decoder_outputs = self.decoder(input_ids=decoder_input_id,
                                       encoder_attention_mask=attention_mask,
                                       encoder_hidden_states=hidden_states)

        decoder_outputs = decoder_outputs['last_hidden_state'] # (batch_size, max_seq_len, hidden_size)
        reconstruction_logits = self.decoder_vocab(decoder_outputs) # (batch_size, max_seq_len, vocab_size)

        return reconstruction_logits

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation
        """
        encoder_outputs = self.encode(input_ids, attention_mask)
        latent_encoded, latent_decoded = self.latent_module(encoder_outputs)
        classification_logits = self.classify(latent_decoded)
        reconstruction_logits = self.decode(input_ids, attention_mask, encoder_outputs, latent_decoded)

        return classification_logits, reconstruction_logits, latent_encoded, latent_decoded

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[: , 1: ] = input_ids[: , : -1].clone()
        shifted_input_ids[: , 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def generate_sequence(self, encoder_outputs: torch.Tensor, latent_decoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Assert batch_size is 1
        assert encoder_outputs.shape[0] == 1

        latent_decoded = latent_decoded.unsqueeze(1).repeat(1, self.max_seq_len, 1) # (batch_size, max_seq_len, hidden_size)
        hidden_states = self.args.aug_encoder_latent_ratio * encoder_outputs + (1 - self.args.aug_encoder_latent_ratio) * latent_decoded # (batch_size, max_seq_len, hidden_size)

        # BART decoder start token id
        decoder_input_ids = torch.tensor([self.bart.config.decoder_start_token_id]).repeat(encoder_outputs.shape[0], 1).to(hidden_states.device) # (batch_size, 1)
        # BART decoder <s> token id
        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([self.bart.config.bos_token_id]).repeat(encoder_outputs.shape[0], 1).to(hidden_states.device)], dim=-1) # (batch_size, 2)

        # Generate sequence
        for i in range(self.max_seq_len - 1): # -1 for <bos>
            decoder_outputs = self.decoder(input_ids=decoder_input_ids,
                                           encoder_attention_mask=attention_mask,
                                           encoder_hidden_states=hidden_states)

            decoder_outputs = decoder_outputs['last_hidden_state'] # (batch_size, current_seq_len, hidden_size)
            decoder_pred_logits = self.decoder_vocab(decoder_outputs) # (batch_size, current_seq_len, vocab_size)
            next_token_logits = decoder_pred_logits[:, -1, :] # (batch_size, vocab_size)

            # Avoid generating <pad> and <s> token
            next_token_logits[:, self.pad_token_id] = -float('inf')
            next_token_logits[:, self.bart.config.bos_token_id] = -float('inf')

            # Apply softmax temperature to logits
            next_token_logits = next_token_logits / self.sampling_temperature
            next_token_prob = F.softmax(next_token_logits, dim=-1) # (batch_size, vocab_size)

            # Get word prediction through predeterminded strategy
            if self.sampling_strategy == 'greedy':
                next_token = torch.argmax(next_token_prob, dim=-1) # (batch_size)
            elif self.sampling_strategy == 'beam':
                raise NotImplementedError
            elif self.sampling_strategy == 'multinomial':
                next_token = torch.multinomial(next_token_prob, num_samples=1).squeeze(1) # (batch_size)
            elif self.sampling_strategy == 'topk':
                topk_prob, topk_idx = torch.topk(next_token_prob, k=self.sampling_topk, dim=-1) # (batch_size, topk)
                topk_prob = topk_prob / torch.sum(topk_prob, dim=-1, keepdim=True) # (batch_size, topk) - normalize to sum to 1
                next_token_idx = torch.multinomial(topk_prob, num_samples=1).squeeze(1) # (batch_size)
                next_token = topk_idx[torch.arange(topk_idx.shape[0]), next_token_idx] # (batch_size)
            elif self.sampling_strategy == 'topp':
                sorted_prob, sorted_idx = torch.sort(next_token_prob, descending=True, dim=-1)
                cumsum_prob = torch.cumsum(sorted_prob, dim=-1)
                sorted_idx_to_remove = cumsum_prob > self.sampling_topp
                sorted_idx_to_remove[..., 1:] = sorted_idx_to_remove[..., :-1].clone()
                sorted_idx_to_remove[..., 0] = 0 # keep at least one token
                sorted_prob[sorted_idx_to_remove] = 0 # set prob to 0 for tokens to remove
                next_token_prob = sorted_prob / torch.sum(sorted_prob, dim=-1, keepdim=True) # normalize to sum to 1
                next_token_idx = torch.multinomial(next_token_prob, num_samples=1).squeeze(1)
                next_token = sorted_idx[torch.arange(sorted_idx.shape[0]), next_token_idx]
            else:
                raise ValueError(f'Invalid sampling strategy: {self.sampling_strategy}')

            # Concatenate next token to decoder_input_ids
            next_token = next_token.unsqueeze(1) # (batch_size, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1) # (batch_size, current_seq_len + 1)
            #print(decoder_input_ids)

            # Break if <eos> token is generated
            if next_token.item() == self.bart.config.eos_token_id:
                break

        return decoder_input_ids