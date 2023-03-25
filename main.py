from engines.utils.logger import Logger
from engines.utils.setup_seed import set_seed
from config import use_cuda, cuda_device, configure, mode
from pprint import pprint
import torch
import json
import time
import os


def fold_check(configures):
    if configures['checkpoints_dir'] == '':
        raise Exception('checkpoints_dir did not set...')

    if not os.path.exists(configures['checkpoints_dir']):
        print('checkpoints fold not found, creating...')
        os.makedirs(configures['checkpoints_dir'])

    if not os.path.exists(configures['checkpoints_dir'] + '/logs'):
        print('log fold not found, creating...')
        os.mkdir(configures['checkpoints_dir'] + '/logs')


if __name__ == '__main__':
    set_seed(configure['seed'])
    fold_check(configure)
    logger = Logger(name='UIE', log_dir=configure['checkpoints_dir'] + '/logs', mode=mode)
    if use_cuda:
        if torch.cuda.is_available():
            if cuda_device == -1:
                device = torch.device('cuda')
            else:
                device = torch.device(f'cuda:{cuda_device}')
        else:
            raise ValueError(
                "'use_cuda' set to True when cuda is unavailable."
                " Make sure CUDA is available or set use_cuda=False."
            )
    else:
        device = 'cpu'

    logger.logger.info(f'device: {device}')
    logger.logger.info(json.dumps(configure, indent=2, ensure_ascii=False))
    if mode == 'train':
        from engines.train import Train
        logger.logger.info('mode: train')
        train = Train(device, logger)
        train.train()
    elif mode == 'test':
        from engines.predict import Predict
        predict = Predict(device, logger)
        predict.predict_test()
    elif mode == 'interactive_predict':
        from engines.predict import Predict
        predict = Predict(device, logger)
        predict.predict_one('warm up')
        while True:
            logger.logger.info('please input a sentence (enter [exit] to exit.)')
            print('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            logger.logger.info('input:{}'.format(str(sentence)))
            start_time = time.time()
            result = predict.predict_one(sentence)
            time_cost = (time.time() - start_time) * 1000
            logger.logger.info('putput:{}, cost {}(ms).'.format(str(result), time_cost))
            pprint(result)
            print('time consumption: %.3f(ms)' % time_cost)
