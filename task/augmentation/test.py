# Standard Library Modules
import os
import sys
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Huggingface Modules
from transformers import AutoTokenizer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.augmentation.model import MainModel
from model.augmentation.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_torch_device, get_huggingface_model_name

def testing(args: argparse.Namespace) -> None:
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
    dataset_test = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'test_processed.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_test.vocab_size
    args.num_classes = dataset_test.num_classes
    args.pad_token_id = dataset_test.pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

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

    # Test - Start testing
    model = model.eval()
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc='Testing')):
        # Test - Get data
        input_ids = data_dicts['input_ids'].to(device)
        attn_masks = data_dicts['attention_mask'].to(device)
        labels = data_dicts['labels'].to(device)

        # Test - Forward pass
        with torch.no_grad():
            encoder_outputs = model.encode(input_ids=input_ids, attention_mask=attn_masks)
            latent_encoded, latent_decoded = model.process_latent_module(encoder_outputs=encoder_outputs)
            classification_logits = model.classify(latent_decoded=latent_decoded)
            classification_probs = F.softmax(classification_logits, dim=-1)
            generated_sequence = model.generate_sequence(encoder_outputs=encoder_outputs,
                                                        latent_decoded=latent_decoded,
                                                        attention_mask=attn_masks)

        # Test - Decode generated sequence using tokenizer
        decoded_sequence = tokenizer.batch_decode(generated_sequence, skip_special_tokens=True)[0]
        tqdm.write(f"Input sequence: {tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]}")
        tqdm.write(f"Classification probs: {classification_probs}")
        tqdm.write(f"Generated sequence: {decoded_sequence}")
        tqdm.write(f"\n")

    write_log(logger, "Testing finished")
