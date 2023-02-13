TASK=augmentation
TASK_DATASET=imdb

python main.py --task=${TASK} --job=preprocessing --task_dataset=${TASK_DATASET} --use_tensorboard=False
python main.py --task=${TASK} --job=training --task_dataset=${TASK_DATASET} --use_tensorboard=True
python main.py --task=${TASK} --job=inference --task_dataset=${TASK_DATASET} --use_tensorboard=False
