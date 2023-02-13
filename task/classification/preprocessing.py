# Standard Library Modules
import os
import sys
import pickle
import argparse
# 3rd-party Modules
import bs4
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_data(dataset_name: str) -> tuple: # (dict, dict, dict, int)
    """
    Load data from huggingface datasets.
    """
    name = dataset_name.lower()

    train_data = {
        'text': [],
        'label': [],
        'soft_label': []
    }
    valid_data = {
        'text': [],
        'label': [],
        'soft_label': []
    }
    test_data = {
        'text': [],
        'label': [],
        'soft_label': []
    }

    if name == 'sst2':
        dataset = load_dataset('glue', 'sst2')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['sentence'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['sentence'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['sentence'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'cola':
        dataset = load_dataset('glue', 'cola')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['sentence'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['sentence'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['sentence'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'imdb':
        dataset = load_dataset('imdb')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        train_data['text'] = train_df['sentence'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['text'] = valid_df['sentence'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['text'] = test_df['sentence'].tolist()
        test_data['label'] = test_df['label'].tolist()

    # Convert integer label to soft label
    for data in [train_data, valid_data]:
        for i, label in enumerate(data['label']):
            soft_label = [0.0] * num_classes
            soft_label[label] = 1.0
            data['soft_label'].append(soft_label)
    # For test data, label is -1 (unknown) so we don't need to convert it to soft label (all zeros)
    for data in [test_data]:
        for i, label in enumerate(data['label']):
            soft_label = [0.0] * num_classes
            # soft_label[label] = 1.0
            data['soft_label'].append(soft_label)

    return train_data, valid_data, test_data, num_classes

def load_augmented_data(path: str) -> dict:
    """
    Load augmented train data from pickle file.
    """
    with open(path, 'rb') as f:
        augmented_data = pickle.load(f)

    return augmented_data

def preprocessing(args: argparse.Namespace) -> None:
    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args.dataset_name)

    # Load augmented data and merge with original data
    if args.use_augmented_data:
        augmented_data = load_augmented_data(args.augmented_data_path)

        # Merge augmented data and original data
        for key in train_data.keys():
            train_data[key].extend(augmented_data[key])

    # Define tokenizer & config
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'valid': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'test': {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'soft_labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        }
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['text'])), desc=f'Preprocessing {split} data'):
            # Get text and label
            text = split_data['text'][idx]
            label = split_data['label'][idx]

            # Remove html tags
            clean_text = bs4.BeautifulSoup(text, 'lxml').text
            # Remove special characters
            clean_text = clean_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
            # Remove multiple spaces
            clean_text = ' '.join(clean_text.split())

            # Tokenize
            tokenized = tokenizer(clean_text, padding='max_length', truncation=True,
                                  max_length=args.max_seq_len, return_tensors='pt')

            # Append data
            data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
            if args.model_type == 'bert':
                data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
            else:
                data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long)) # Cross Entropy Loss
            data_dict[split]['soft_labels'].append(torch.tensor(split_data['soft_label'][idx], dtype=torch.float)) # Soft Cross Entropy Loss

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
