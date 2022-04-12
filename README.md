# SSTOD
This is the code and data for the ACL 2022 paper "A Slot Is Not Built in One Utterance: Spoken Language Dialogs with Sub-Slots". [arxiv](https://arxiv.org/pdf/2203.10759.pdf)

## Abstract
A slot value might be provided segment by segment over multiple-turn interactions in a dialog, especially for some important information such as phone numbers and names. It is a common phenomenon in daily life, but little attention has been paid to it in previous work. To fill the gap, this paper defines a new task named Sub-Slot based Task-Oriented Dialog (SSTOD) and builds a Chinese dialog dataset SSD for boosting research on SSTOD. The dataset includes a total of 40K dialogs and 500K utterances from four different domains: Chinese names, phone numbers, ID numbers and license plate numbers. The data is well annotated with sub-slot values, slot values, dialog states and actions. We find some new linguistic phenomena and interactive manners in SSTOD which raise critical challenges of building dialog agents for the task. We test three state-of-the-art dialog models on SSTOD and find they cannot handle the task well on any of the four domains. We also investigate an improved model by involving slot knowledge in a plug-in manner. More work should be done to meet the new challenges raised from SSTOD which widely exists in real-life applications. 

## SSD dataset
The SSD dataset is available at [TIANCHI](https://tianchi.aliyun.com/dataset/dataDetail?dataId=125708).

## Requirements
- CUDA 10.1
- python 3.6
- pytorch 1.8.0
- pypinyin
- spaCy
- transformers 4.10.2

## Training
Our implementation supports training on CPU or a single GPU.
```
python train.py --mode train --exp_path ./output/UBAR_plus --data_dir $DATA_PATH --device cuda:0
```

## Evaluation
Evaluation with the ground-truth knowledge predict results
```
python train.py --mode test --eval_load_path $MODEL_PATH --data_dir $DATA_PATH --use_true_curr_kdpn True --use_true_curr_kp True --device cuda:0
```

Evaluation with the ground-truth knowledge select results
```
python train.py --mode test --eval_load_path $MODEL_PATH --data_dir $DATA_PATH --use_true_db_pointer True --device cuda:0
```

End-to-end evaluation
```
python train.py --mode test --eval_load_path $MODEL_PATH --data_dir $DATA_PATH --device cuda:0
```


## Acknowledge
This code is adapted and modified upon the released code of previous AAAI 2021 paper "UBAR: Towards Fully End-to-End Task-Oriented Dialog System with GPT-2". 