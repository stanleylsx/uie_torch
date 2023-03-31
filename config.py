# -*- coding: utf-8 -*-
# @Time : 2021/07/15 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm

# 模式
# train:                训练UIE
# interactive_predict:  交互模式
# test:                 跑测试集
# export_torch:         转torch
# export_onnx:          转onnx
# convert_label_studio: 把label studio的数据转成数据集
mode = 'convert_label_studio'

# 使用GPU设备
use_cuda = True
cuda_device = -1

show_bar = True

configure = {
    # prompt schema
    'schema': ['姓名', '地址', '小区', '单元房号', '手机号'],
    'model_type': 'uie-base',
    # 训练数据集
    'train_file': 'datasets/train.txt',
    # 验证数据集
    'val_file': 'datasets/dev.txt',
    # 测试数据集
    'test_file': 'datasets/dev.txt',
    # 引擎onnx或者pytorch
    'engine': 'onnx',
    # 模型语言
    'schema_lang': 'zh',
    # 是否多语言
    'multilingual': False,
    # 没有验证集时，从训练集抽取验证集比例
    'validation_rate': 0.1,
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoint',
    # 句子最大长度
    'max_sequence_length': 512,
    # 判决阈值
    'position_prob': 0.5,
    # 分割句子
    'split_sentence': True,
    # 随机种子
    'seed': 1000,
    # 使用fp16
    'use_fp16': False,
    # 微调阶段的epoch
    'epoch': 30,
    # 微调阶段的batch_size
    'batch_size': 4,
    # 微调阶段的学习率
    'learning_rate': 1e-5,
    # 微调阶段每print_per_batch打印
    'print_per_batch': 10,
    # The interval steps to evaluate model performance.
    'valid_steps': 100,
    # 是否提前结束微调
    'is_early_stop': True,
    'patience': 2,
    # Max number of saved model. Best model and early-stopping model is not included.
    'max_model_num': 1,
    'label_studio_file': 'datasets/project.json',
    'data_splits': [0.8, 0.1, 0.1],
    'task_type': 'ext',
    'negative_ratio': 5
}
