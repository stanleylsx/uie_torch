# Copyright (c) 2022 Heiheiyoyo. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil
import sys
import time
import os
import torch
from torch.utils.data import DataLoader
from engines.utils.tqdm_util import tqdm, logging_redirect_tqdm
from engines.data import IEDataset
from engines.models.uie import UIE, UIEM
from engines.utils.span_evaluator import SpanEvaluator
from engines.utils.early_stop import EarlyStopping
from config import show_bar, configure, mode


class Train:
    def __init__(self, device, logger):
        self.logger = logger
        self.device = device
        self.train_path = configure['train_file']
        self.dev_path = configure['val_file']
        self.max_seq_len = configure['max_sequence_length']
        self.batch_size = configure['batch_size']
        self.learning_rate = configure['learning_rate']
        self.checkpoints_dir = configure['checkpoints_dir']
        self.num_epochs = configure['epoch']
        self.logging_steps = configure['print_per_batch']
        self.valid_steps = configure['valid_steps']
        self.max_model_num = configure['max_model_num']
        self.multilingual = configure['multilingual']
        model_type = configure['model_type']
        self.model_path = os.path.join(model_type, 'torch')

        if self.multilingual:
            from engines.utils.tokenizer import ErnieMTokenizerFast
            self.tokenizer = ErnieMTokenizerFast.from_pretrained(self.model_path)
        else:
            from transformers import BertTokenizerFast
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)

        self.metric = SpanEvaluator()
        self.criterion = torch.nn.functional.binary_cross_entropy
        if mode == 'train' and configure['is_early_stop']:
            self.early_stopping_save_dir = os.path.join(self.checkpoints_dir, 'early_stopping')
            if not os.path.exists(self.early_stopping_save_dir):
                os.makedirs(self.early_stopping_save_dir)
            if show_bar:
                def trace_func(*args, **kwargs):
                    with logging_redirect_tqdm([self.logger.logger]):
                        self.logger.info(*args, **kwargs)
            else:
                trace_func = self.logger.info
            self.early_stopping = EarlyStopping(patience=configure['patience'], verbose=True, trace_func=trace_func,
                                           save_dir=self.early_stopping_save_dir)

    def train(self):
        if self.multilingual:
            model = UIEM.from_pretrained(self.model_path).to(self.device)
        else:
            model = UIE.from_pretrained(self.model_path).to(self.device)

        train_ds = IEDataset(self.train_path, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
        dev_ds = IEDataset(self.dev_path, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
        train_data_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        dev_data_loader = DataLoader(dev_ds, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(lr=self.learning_rate, params=model.parameters())
        loss_list = []
        loss_sum = 0
        loss_num = 0
        global_step = 0
        best_f1 = 0
        tic_train = time.time()
        epoch_iterator = range(1, self.num_epochs + 1)
        train_postfix_info = None
        if show_bar:
            train_postfix_info = {'loss': 'unknown'}
            epoch_iterator = tqdm(epoch_iterator, desc='Training', unit='epoch')
        for epoch in epoch_iterator:
            train_data_iterator = train_data_loader
            if show_bar:
                train_data_iterator = tqdm(train_data_iterator, desc=f'Training Epoch {epoch}', unit='batch')
                train_data_iterator.set_postfix(train_postfix_info)
            for batch in train_data_iterator:
                if show_bar:
                    epoch_iterator.refresh()
                input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                att_mask = att_mask.to(self.device)
                start_ids = start_ids.to(self.device)
                end_ids = end_ids.to(self.device)
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=att_mask)
                start_prob, end_prob = outputs[0], outputs[1]
                start_ids = start_ids.type(torch.float32)
                end_ids = end_ids.type(torch.float32)
                loss_start = self.criterion(start_prob, start_ids)
                loss_end = self.criterion(end_prob, end_ids)
                loss = (loss_start + loss_end) / 2.0
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_list.append(float(loss))
                loss_sum += float(loss)
                loss_num += 1

                if show_bar:
                    loss_avg = loss_sum / loss_num
                    train_postfix_info.update({'loss': f'{loss_avg:.5f}'})
                    train_data_iterator.set_postfix(train_postfix_info)

                global_step += 1
                if global_step % self.logging_steps == 0:
                    time_diff = time.time() - tic_train
                    loss_avg = loss_sum / loss_num

                    if show_bar:
                        with logging_redirect_tqdm([self.logger.logger]):
                            self.logger.info(
                                'global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s' % (
                                    global_step, epoch, loss_avg, self.logging_steps / time_diff))
                    else:
                        self.logger.info('global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s'
                                    % (global_step, epoch, loss_avg, self.logging_steps / time_diff))
                    tic_train = time.time()

                if global_step % self.valid_steps == 0:
                    save_dir = os.path.join(self.checkpoints_dir, 'model_%d' % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    model_to_save = model
                    model_to_save.save_pretrained(save_dir)
                    if self.max_model_num:
                        model_to_delete = global_step - self.max_model_num * self.valid_steps
                        model_to_delete_path = os.path.join(self.checkpoints_dir, 'model_%d' % model_to_delete)
                        if model_to_delete > 0 and os.path.exists(model_to_delete_path):
                            shutil.rmtree(model_to_delete_path)
                    dev_loss_avg, precision, recall, f1 = self.evaluate(model, data_loader=dev_data_loader)
                    if show_bar:
                        train_postfix_info.update({'F1': f'{f1:.3f}', 'dev loss': f'{dev_loss_avg:.5f}'})
                        train_data_iterator.set_postfix(train_postfix_info)
                        with logging_redirect_tqdm([self.logger.logger]):
                            self.logger.info('Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f'
                                        % (precision, recall, f1, dev_loss_avg))
                    else:
                        self.logger.info('Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f'
                                    % (precision, recall, f1, dev_loss_avg))
                    # Save model which has best F1
                    if f1 > best_f1:
                        if show_bar:
                            with logging_redirect_tqdm([self.logger.logger]):
                                self.logger.info(f'best F1 performance has been updated: {best_f1:.5f} --> {f1:.5f}')
                        else:
                            self.logger.info(f'best F1 performance has been updated: {best_f1:.5f} --> {f1:.5f}')
                        best_f1 = f1
                        save_dir = os.path.join(self.checkpoints_dir, 'model_best')
                        model_to_save = model
                        model_to_save.save_pretrained(save_dir)
                    tic_train = time.time()

            if configure['is_early_stop']:
                dev_loss_avg, precision, recall, f1 = self.evaluate(model, data_loader=dev_data_loader)
                if show_bar:
                    train_postfix_info.update({'F1': f'{f1:.3f}', 'dev loss': f'{dev_loss_avg:.5f}'})
                    train_data_iterator.set_postfix(train_postfix_info)
                    with logging_redirect_tqdm([self.logger.logger]):
                        self.logger.info('Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f'
                                    % (precision, recall, f1, dev_loss_avg))
                else:
                    self.logger.info('Evaluation precision: %.5f, recall: %.5f, F1: %.5f, dev loss: %.5f'
                                % (precision, recall, f1, dev_loss_avg))

                # Early Stopping
                self.early_stopping(dev_loss_avg, model)
                if self.early_stopping.early_stop:
                    if show_bar:
                        with logging_redirect_tqdm([self.logger.logger]):
                            self.logger.info('Early stopping')
                    else:
                        self.logger.info('Early stopping')
                    sys.exit(0)

    @torch.no_grad()
    def evaluate(self, model, data_loader, return_loss=True):
        """
        Given a dataset, it will evaluate model and computes the metric.
        Args:
            model(obj:`torch.nn.Module`): A model to classify texts.
            data_loader(obj:`torch.utils.data.DataLoader`): The dataset loader which generates batches.
            return_loss
        """
        model.eval()
        self.metric.reset()
        loss_list = []
        loss_sum = 0
        loss_num = 0
        if show_bar:
            data_loader = tqdm(data_loader, desc='Evaluating', unit='batch')
        for batch in data_loader:
            input_ids, token_type_ids, att_mask, start_ids, end_ids = batch
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            att_mask = att_mask.to(self.device)
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=att_mask)
            start_prob, end_prob = outputs[0], outputs[1]
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
            start_ids = start_ids.type(torch.float32)
            end_ids = end_ids.type(torch.float32)

            if return_loss:
                # Calculate loss
                loss_start = self.criterion(start_prob, start_ids)
                loss_end = self.criterion(end_prob, end_ids)
                loss = (loss_start + loss_end) / 2.0
                loss = float(loss)
                loss_list.append(loss)
                loss_sum += loss
                loss_num += 1
                if show_bar:
                    data_loader.set_postfix({'dev loss': f'{loss_sum / loss_num:.5f}'})

            # Calculate metric
            num_correct, num_infer, num_label = self.metric.compute(start_prob, end_prob, start_ids, end_ids)
            self.metric.update(num_correct, num_infer, num_label)
        precision, recall, f1 = self.metric.accumulate()
        model.train()
        if return_loss:
            loss_avg = sum(loss_list) / len(loss_list)
            return loss_avg, precision, recall, f1
        else:
            return precision, recall, f1
