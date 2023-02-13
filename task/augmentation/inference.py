# Standard Library Modules
import os
import sys
import pickle
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.augmentation.model import MainModel
from model.augmentation.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_torch_device, get_huggingface_model_name, check_path

def inference(args: argparse.Namespace) -> None:
    """
    Inference function for augmentation task
    This function implements the inference process of the augmentation task.
    Core logic of the augmentation task is the gradient modification of encoder_output.
    """
    device = get_torch_device(args.device)

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading data")
    dataset_train = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'train_processed.pkl'))

    dataloader_train = DataLoader(dataset_train, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
    args.vocab_size = dataset_train.vocab_size
    args.num_classes = dataset_train.num_classes
    args.pad_token_id = dataset_train.pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_train)} / {len(dataloader_train)}")

    # Get model instance
    write_log(logger, "Building model")
    model = MainModel(args).to(device)

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, 'final_model.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Get tokenizer for decoding
    huggingface_model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)

    # Get classification loss function for gradient calculation
    cls_loss = nn.CrossEntropyLoss()

    # Define augmented data dictionary
    augmented_data_dict = {
        'text': [],
        'label': [],
        'soft_label': []
    }

    # Inference - Start augmentation for train data
    model = model.eval()
    for iter_idx, data_dicts in enumerate(tqdm(dataloader_train, total=len(dataloader_train), desc='Augmentation')):
        # Infernece - Get data
        input_ids = data_dicts['input_ids'].to(device)
        attn_masks = data_dicts['attention_mask'].to(device)

        # Inference - Forward pass - Get classification result
        original_encoder_outputs = model.encode(input_ids=input_ids, attention_mask=attn_masks)
        original_latent_encoded, original_latent_decoded = model.process_latent_module(encoder_outputs=original_encoder_outputs)
        original_classification_logits = model.classify(latent_decoded=original_latent_decoded)
        original_classification_probs = F.softmax(original_classification_logits, dim=-1)
        #original_encoder_outputs.retain_grad() # Retain encoder_output gradient for gradient modification
        original_latent_decoded.retain_grad() # Retain latent_decoded gradient for gradient modification

        # Inference - Get gradient of encoder_output using classification result
        cls_loss_value = cls_loss(original_classification_logits, data_dicts['labels'].to(device))
        cls_loss_value.backward()
        #encoder_output_grad = original_encoder_outputs.grad
        latent_decoded_grad = original_latent_decoded.grad

        tqdm.write(f"Original sequence: {tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]}")
        tqdm.write(f"Original Classification Probs: {original_classification_probs[0].cpu().detach()}")

        # STARTEGY 1: Gradient modification from the original encoder_output
        # Inference - Modify encoder_output using gradient and generate augmented sequence
        for grad_eps in [2, 3, 5, 10, 20, 30, 50, 100]:
            # Inference - Modify encoder_output using gradient
            #modified_encoder_outputs = original_encoder_outputs + encoder_output_grad * grad_eps
            modified_latent_decoded = original_latent_decoded + latent_decoded_grad * grad_eps
            #modified_latent_encoded, modified_latent_decoded = model.process_latent_module(encoder_outputs=modified_encoder_outputs)
            modified_classification_logits = model.classify(latent_decoded=modified_latent_decoded)
            modified_classification_probs = F.softmax(modified_classification_logits, dim=-1)

            # Inference - Pass if classification label has changed after gradient modification - prevent label flip
            if torch.argmax(modified_classification_probs, dim=-1) != torch.argmax(original_classification_logits, dim=-1):
                continue

            augmented_sequence = model.generate_sequence(encoder_outputs=original_encoder_outputs,
                                                         latent_decoded=modified_latent_decoded,
                                                         attention_mask=attn_masks)

            # Inference - Decode augmented sequence using tokenizer
            decoded_augmented_sequence = tokenizer.batch_decode(augmented_sequence, skip_special_tokens=True)[0]

            # Inference - Classify augmented sequence
            with torch.no_grad():
                augmented_tokenized = tokenizer(decoded_augmented_sequence, padding='max_length', truncation=True,
                                                max_length=args.max_seq_len, return_tensors='pt')
                augmented_encoder_output = model.encode(input_ids=augmented_tokenized['input_ids'].to(device),
                                                        attention_mask=augmented_tokenized['attention_mask'].to(device))
                augmented_latent_encoded, augmented_latent_decoded = model.process_latent_module(encoder_outputs=augmented_encoder_output)
                augmented_classification_logits = model.classify(latent_decoded=augmented_latent_decoded)
                augmented_classification_probs = F.softmax(augmented_classification_logits, dim=-1)

            # Inference - Print result
            tqdm.write(f"Modified classification vector at eps {grad_eps}: {modified_classification_probs[0].cpu().detach()}")
            tqdm.write(f"Augmented sequence at eps {grad_eps}: {decoded_augmented_sequence}")
            tqdm.write(f"Classification result of augmented sequence: {augmented_classification_probs[0].cpu().detach()}\n")

            # Inference - Append augmented data to dictionary
            augmented_data_dict['text'].append(decoded_augmented_sequence)
            augmented_data_dict['label'].append(data_dicts['labels'].item())
            augmented_data_dict['soft_label'].append(modified_classification_probs[0].cpu().detach())

        """
        # STRATEGY 2: Iterative gradient modification from the original encoder_output
        # Inference - Modify encoder_output using gradient and generate augmented sequence
        modified_encoder_outputs = original_encoder_outputs
        gradient_sign = 1
        gradient_eps = 20
        for idx in range(30):
            # Inference - Modify encoder_output using gradient
            modified_encoder_outputs = modified_encoder_outputs + (gradient_sign * gradient_eps) * encoder_output_grad
            modified_latent_encoded, modified_latent_decoded = model.process_latent_module(encoder_outputs=modified_encoder_outputs)
            modified_classification_logits = model.classify(latent_decoded=modified_latent_decoded)
            modified_classification_probs = F.softmax(modified_classification_logits, dim=-1)

            "
            # Inference - Prevent label flip
            if torch.argmax(modified_classification_probs, dim=-1) != torch.argmax(original_classification_probs, dim=-1):
                gradient_sign *= -1

                tqdm.write(f"Label-Flipped Classification Probs at iter {idx}: {modified_classification_probs[0].cpu().detach()}")
                #tqdm.write(f"Label flip detected. Stop gradient modification")
                tqdm.write(f"Label flip detected. Change gradient sign to {gradient_sign}\n")
                continue

            # Inference - Prevent overshooting
            if torch.argmax(modified_classification_probs, dim=-1) == torch.argmax(original_classification_probs, dim=-1) and \
                torch.max(modified_classification_probs, dim=-1)[0] > torch.max(original_classification_probs, dim=-1)[0]:

                gradient_sign *= -1
                gradient_eps *= 0.5
                modified_encoder_outputs = original_encoder_outputs
                tqdm.write(f"Overshooting detected. Change gradient sign to {gradient_sign} and eps to {gradient_eps}")
                tqdm.write(f"Overshooted Classification Probs at iter {idx}: {modified_classification_probs[0].cpu().detach()}")
                tqdm.write(f"Reset modified_encoder_outputs to original_encoder_outputs\n")
                continue
            "

            # Inference - Generate augmented sequence - no gradient calculation needed
            with torch.no_grad():
                augmented_sequence = model.generate_sequence(encoder_outputs=modified_encoder_outputs,
                                                            latent_decoded=modified_latent_decoded,
                                                            attention_mask=attn_masks)

            # Inference - Decode augmented sequence using tokenizer
            decoded_augmented_sequence = tokenizer.batch_decode(augmented_sequence, skip_special_tokens=True)[0]

            # Inference - Classify augmented sequence
            with torch.no_grad():
                augmented_tokenized = tokenizer(decoded_augmented_sequence, padding='max_length', truncation=True,
                                                max_length=args.max_seq_len, return_tensors='pt')
                augmented_encoder_output = model.encode(input_ids=augmented_tokenized['input_ids'].to(device),
                                                        attention_mask=augmented_tokenized['attention_mask'].to(device))
                augmented_latent_encoded, augmented_latent_decoded = model.process_latent_module(encoder_outputs=augmented_encoder_output)
                augmented_classification_logits = model.classify(latent_decoded=augmented_latent_decoded)
                augmented_classification_probs = F.softmax(augmented_classification_logits, dim=-1)

            # Inference - Print result
            tqdm.write(f"Modified classification vector at iter {idx}: {modified_classification_probs[0].cpu().detach()}")
            tqdm.write(f"Augmented sequence at iter {idx}: {decoded_augmented_sequence}")
            tqdm.write(f"Classification result of augmented sequence: {augmented_classification_probs[0].cpu().detach()}\n")

            # Inference - Append augmented data to dictionary
            augmented_data_dict['text'].append(decoded_augmented_sequence)
            augmented_data_dict['label'].append(data_dicts['labels'].item())
            augmented_data_dict['soft_label'].append(modified_classification_probs[0].cpu().detach())
        tqdm.write("\n")
        """

        # Inference - Save augmented data as pickle file to result_path
        save_path = os.path.join(args.result_path, args.task, args.task_dataset, args.model_type)
        check_path(save_path)

        with open(os.path.join(save_path, f"augmented_train.pkl"), "wb") as f:
            pickle.dump(augmented_data_dict, f)
