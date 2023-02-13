# Standard Library Modules
import os
import argparse
# Custom Modules
from utils.utils import parse_bool

class ArgParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.user_name = os.getlogin()
        self.proj_name = 'gradient_text_aug'

        # Task arguments
        task_list = ['augmentation', 'classification']
        self.parser.add_argument('--task', type=str, choices=task_list, default='augmentation',
                                 help='Task to do; Must be given.')
        job_list = ['preprocessing', 'training', 'resume_training', 'testing', 'inference', 'augmenting']
        self.parser.add_argument('--job', type=str, choices=job_list, default='training',
                                 help='Job to do; Must be given.')
        dataset_list = ['imdb', 'sst2', 'cola']
        self.parser.add_argument('--task_dataset', type=str, choices=dataset_list, default='cola',
                            help='Dataset for the task; Must be given.')

        # Path arguments
        self.parser.add_argument('--data_path', type=str, default='/HDD/dataset',
                                 help='Path to the raw dataset before preprocessing.')
        self.parser.add_argument('--preprocess_path', type=str, default=f'/HDD/{self.user_name}/preprocessed',
                                 help='Path to the preprocessed dataset.')
        self.parser.add_argument('--model_path', type=str, default=f'/HDD/{self.user_name}/model_final/{self.proj_name}',
                                 help='Path to the model after training.')
        self.parser.add_argument('--checkpoint_path', type=str, default=f'/HDD/{self.user_name}/model_checkpoint/{self.proj_name}')
        self.parser.add_argument('--result_path', type=str, default=f'/HDD/{self.user_name}/results/{self.proj_name}',
                                 help='Path to the result after testing.')
        self.parser.add_argument('--log_path', type=str, default=f'/HDD/{self.user_name}/tensorboard_log/{self.proj_name}',
                                 help='Path to the tensorboard log file.')

        # Model - Basic arguments
        self.parser.add_argument('--proj_name', type=str, default='gradient_text_aug',
                                 help='Name of the project.')
        model_type_list = ['bart', 'bert', 'roberta', 't5']
        self.parser.add_argument('--model_type', type=str, choices=model_type_list, default='bart',
                                 help='Type of the classification model to use.')
        self.parser.add_argument('--model_ispretrained', type=parse_bool, default=True,
                                 help='Whether to use pretrained model; Default is True')
        self.parser.add_argument('--min_seq_len', type=int, default=4,
                                 help='Minimum sequence length of the input; Default is 4')
        self.parser.add_argument('--max_seq_len', type=int, default=300,
                                 help='Maximum sequence length of the input; Default is 300')
        self.parser.add_argument('--dropout_rate', type=float, default=0.3,
                                 help='Dropout rate of the model; Default is 0.3')

        # Model - Size arguments
        self.parser.add_argument('--embed_size', type=int, default=768, # Will be automatically specified by the model type
                                 help='Embedding size of the model; Default is 768')
        self.parser.add_argument('--hidden_size', type=int, default=768, # Will be automatically specified by the model type
                                 help='Hidden size of the model; Default is 768')
        self.parser.add_argument('--latent_size', type=int, default=32,
                                 help='Latent size of the model; Default is 32')

        # Model - Optimizer & Scheduler arguments
        optim_list = ['SGD', 'AdaDelta', 'Adam', 'AdamW']
        scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
        self.parser.add_argument('--optimizer', type=str, choices=optim_list, default='AdamW',
                                 help="Optimizer to use; Default is Adam")
        self.parser.add_argument('--cls_scheduler', type=str, choices=scheduler_list, default='CosineAnnealingLR',
                                 help="Scheduler to use for classification; Default is CosineAnnealingLR")
        self.parser.add_argument('--aug_scheduler', type=str, choices=scheduler_list, default='CosineAnnealingLR',
                                 help="Scheduler to use for reconstruction; Default is CosineAnnealingLR")

        # Training arguments 1
        self.parser.add_argument('--cls_num_epochs', type=int, default=20,
                                 help='Training epochs for classifier; Default is 20')
        self.parser.add_argument('--cls_learning_rate', type=float, default=1e-5,
                                 help='Learning rate of optimizer for classifier; Default is 1e-5')
        self.parser.add_argument('--aug_num_epochs', type=int, default=100,
                                 help='Training epochs for augmenter; Default is 20')
        self.parser.add_argument('--aug_learning_rate', type=float, default=5e-5,
                                 help='Learning rate of optimizer for augmenter; Default is 5e-5')
        self.parser.add_argument('--aug_encoder_latent_ratio', type=float, default=1.0,
                                 help='Ratio of encoder hidden state versus latent vector; Default is 1.0 - No latent vector')

        # Training arguments 2
        self.parser.add_argument('--num_workers', type=int, default=2,
                                 help='Num CPU Workers; Default is 2')
        self.parser.add_argument('--batch_size', type=int, default=16,
                                 help='Batch size; Default is 32')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5,
                                 help='Weight decay; Default is 5e-4; If 0, no weight decay')
        self.parser.add_argument('--clip_grad_norm', type=int, default=5,
                                 help='Gradient clipping norm; Default is 5')
        self.parser.add_argument('--label_smoothing_eps', type=float, default=0.05,
                                 help='Label smoothing epsilon; Default is 0.05')
        self.parser.add_argument('--early_stopping_patience', type=int, default=10,
                                 help='Early stopping patience; No early stopping if None; Default is 10')
        self.parser.add_argument('--train_valid_split', type=float, default=0.2,
                                 help='Train/Valid split ratio; Default is 0.2')
        objective_list = ['loss', 'accuracy']
        self.parser.add_argument('--optimize_objective', type=str, choices=objective_list, default='loss',
                                 help='Objective to optimize; Default is loss')
        self.parser.add_argument('--use_augmented_data', type=parse_bool, default=False,
                                 help='Whether to use augmented data; Default is False')

        # Testing/Inference arguments
        self.parser.add_argument('--test_batch_size', default=1, type=int,
                                 help='Batch size for test; Default is 1')
        strategy_list = ['greedy', 'beam', 'multinomial', 'topk', 'topp']
        self.parser.add_argument('--sampling_strategy', type=str, choices=strategy_list, default='topk',
                                 help='Sampling strategy for test; Default is greedy')
        self.parser.add_argument('--sampling_temperature', default=5.0, type=float,
                                 help='Temperature for multinomial sampling; Default is 5.0')
        self.parser.add_argument('--sampling_topk', default=5, type=int,
                                 help='Top-k sampling; Default is 5')
        self.parser.add_argument('--sampling_topp', default=0.9, type=float,
                                 help='Top-p sampling; Default is 0.9')
        """
        self.parser.add_argument('--beam_size', default=5, type=int,
                                 help='Beam search size; Default is 5')
        self.parser.add_argument('--beam_alpha', default=0.7, type=float,
                                 help='Beam search length normalization; Default is 0.7')
        self.parser.add_argument('--repetition_penalty', default=1.3, type=float,
                                 help='Beam search repetition penalty term; Default is 1.3')
        """

        # Other arguments - Device, Seed, Logging, etc.
        self.parser.add_argument('--device', type=str, default='cuda:0',
                                 help='Device to use for training; Default is cuda')
        self.parser.add_argument('--seed', type=int, default=2023,
                                 help='Random seed; Default is 2023')
        self.parser.add_argument('--use_tensorboard', type=parse_bool, default=True,
                                 help='Using tensorboard; Default is True')
        self.parser.add_argument('--log_freq', default=500, type=int,
                                 help='Logging frequency; Default is 500')

    def get_args(self):
        return self.parser.parse_args()
