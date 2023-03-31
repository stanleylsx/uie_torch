from engines.train import Train
from engines.data import IEDataset
from torch.utils.data import DataLoader
from engines.utils.text_utils import dbc2sbc, cut_chinese_sent
from engines.utils.span_evaluator import get_bool_ids_greater_than, get_span
from engines.data import unify_prompt_name, get_relation_type_dict, IEMapDataset, get_id_and_prob
from config import configure, mode
from itertools import chain
import numpy as np
import torch
import math
import os


class Predict:
    def __init__(self, device, logger):
        self.logger = logger
        self.device = device
        self.max_seq_len = configure['max_sequence_length']
        self.batch_size = configure['batch_size']
        self.position_prob = configure['position_prob']
        self.engine = configure['engine']
        self.schema = configure['schema']
        checkpoints_dir = configure['checkpoints_dir']
        model_type = configure['model_type']
        if configure['is_early_stop']:
            self.model_path = os.path.join(checkpoints_dir, 'early_stopping')
        else:
            self.model_path = os.path.join(checkpoints_dir, 'best_model')
        if not os.path.exists(self.model_path):
            from engines.utils.convert import check_model, extract_and_convert
            model_path = os.path.join(model_type, 'torch')
            if not os.path.exists(model_type):
                check_model(model_type)
                extract_and_convert(model_type, model_path)
            self.model_path = model_path
        assert self.engine in ['pytorch', 'onnx'], 'engine must be pytorch or onnx!'
        self.multilingual = configure['multilingual']

        token_path = os.path.join(model_type, 'torch')
        if self.multilingual:
            from engines.utils.tokenizer import ErnieMTokenizerFast
            self.tokenizer = ErnieMTokenizerFast.from_pretrained(token_path)
        else:
            from transformers import BertTokenizerFast
            self.tokenizer = BertTokenizerFast.from_pretrained(token_path)

        self.split_sentence = configure['split_sentence']
        use_fp16 = configure['use_fp16']
        schema_lang = configure['schema_lang']
        self.train = Train(device, logger)
        self.is_en = True if schema_lang == 'en' else False
        self._schema_tree = None

        if mode == 'export_onnx' and self.engine != 'pytorch':
            raise Exception('please make sure pytorch model on your files when you export onnx!')

        if self.engine == 'pytorch':
            from engines.models.uie import UIE, UIEM
            logger.logger.info('>>> [PyTorchInferBackend] Creating Engine ...')
            if self.multilingual:
                self.model = UIEM.from_pretrained(self.model_path)
            else:
                self.model = UIE.from_pretrained(self.model_path)
            self.model.eval()
            if use_fp16:
                logger.logger.info('>>> [PyTorchInferBackend] Use FP16 to inference ...')
                self.model = self.model.half()
            self.model = self.model.to(device)
            logger.logger.info('>>> [PyTorchInferBackend] Engine Created ...')
        if self.engine == 'onnx':
            if os.path.exists(os.path.join(self.model_path, 'pytorch_model.bin')) \
                    and not os.path.exists(os.path.join(self.model_path, 'inference.onnx')):
                from engines.models.uie import UIE, UIEM
                if self.multilingual:
                    self.model = UIEM.from_pretrained(self.model_path)
                else:
                    self.model = UIE.from_pretrained(self.model_path)
                logger.info('Converting to the inference model cost a little time.')
                save_path = self.export_onnx()
                logger.info('The inference model save in the path:{}'.format(save_path))
                del self.model
            from onnxruntime import InferenceSession, SessionOptions
            logger.logger.info('>>> [ONNXInferBackend] Creating Engine ...')
            onnx_model = float_onnx_file = os.path.join(self.model_path, 'inference.onnx')
            if not os.path.exists(onnx_model):
                raise OSError(f'{onnx_model} not exists!')
            if device == 'gpu':
                providers = ['CUDAExecutionProvider']
                logger.logger.info('>>> [ONNXInferBackend] Use GPU to inference ...')
                if use_fp16:
                    logger.logger.info('>>> [ONNXInferBackend] Use FP16 to inference ...')
                    from onnxconverter_common import float16
                    import onnx
                    fp16_model_file = os.path.join(self.model_path, 'fp16_model.onnx')
                    onnx_model = onnx.load_model(float_onnx_file)
                    trans_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
                    onnx.save_model(trans_model, fp16_model_file)
                    onnx_model = fp16_model_file
            else:
                providers = ['CPUExecutionProvider']
                logger.logger.info('>>> [ONNXInferBackend] Use CPU to inference ...')

            sess_options = SessionOptions()
            self.predictor = InferenceSession(onnx_model, sess_options=sess_options, providers=providers)
            if device == 'gpu':
                try:
                    assert 'CUDAExecutionProvider' in self.predictor.get_providers()
                except AssertionError:
                    raise AssertionError(
                        f'The environment for GPU inference is not set properly. '
                        'A possible cause is that you had installed both onnxruntime and onnxruntime-gpu. '
                        'Please run the following commands to reinstall: \n '
                        '1) pip uninstall -y onnxruntime onnxruntime-gpu \n 2) pip install onnxruntime-gpu'
                    )
            logger.logger.info('>>> [InferBackend] Engine Created ...')

        self.debug = False

    def predict_one(self, inputs):
        self.set_schema()
        texts = inputs
        if isinstance(texts, str):
            texts = [texts]
        results = self._multi_stage_predict(texts)
        return results

    def inference_backend(self, input_dict):
        if self.engine == 'pytorch':
            for input_name, input_value in input_dict.items():
                input_value = torch.LongTensor(input_value).to(self.device)
                input_dict[input_name] = input_value

            outputs = self.model(**input_dict)
            start_prob, end_prob = outputs[0], outputs[1]
            start_prob, end_prob = start_prob.cpu(), end_prob.cpu()
            start_prob = start_prob.detach().numpy()
            end_prob = end_prob.detach().numpy()
            return start_prob, end_prob
        else:
            result = self.predictor.run(None, dict(input_dict))
            return result

    def predict_test(self):
        test_path = configure['test_file']
        if test_path == '':
            self.logger.info('test file does not exist!')
            return
        test_ds = IEDataset(test_path, tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
        class_dict = {}
        relation_data = []
        relation_type_dict = {}
        if self.debug:
            for data in test_ds.dataset:
                class_name = unify_prompt_name(data['prompt'])
                # Only positive examples are evaluated in debug mode
                if len(data['result_list']) != 0:
                    if "的" not in data['prompt']:
                        class_dict.setdefault(class_name, []).append(data)
                    else:
                        relation_data.append((data['prompt'], data))
            relation_type_dict = get_relation_type_dict(relation_data)
        else:
            class_dict['all_classes'] = test_ds

        for key in class_dict.keys():
            if self.debug:
                test_ds = IEMapDataset(class_dict[key], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
            else:
                test_ds = class_dict[key]

            test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
            precision, recall, f1 = self.train.evaluate(self.model, test_data_loader, return_loss=False)
            self.logger.info('-----------------------------')
            self.logger.info('Class Name: %s' % key)
            self.logger.info('Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f' % (precision, recall, f1))

        if self.debug and len(relation_type_dict.keys()) != 0:
            for key in relation_type_dict.keys():
                test_ds = IEMapDataset(relation_type_dict[key], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)
                test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
                precision, recall, f1 = self.train.evaluate(self.model, test_data_loader, return_loss=False)
                self.logger.info('-----------------------------')
                self.logger.info('Class Name: X的%s' % key)
                self.logger.info('Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f' % (precision, recall, f1))

    def set_schema(self):
        if isinstance(self.schema, dict) or isinstance(self.schema, str):
            self.schema = [self.schema]
        self._schema_tree = self._build_tree(self.schema)

    def _multi_stage_predict(self, data):
        """
        Traversal the schema tree and do multi-stage prediction.
        Args:
            data (list): a list of strings
        Returns:
            list: a list of predictions, where the list's length
                equals to the length of `data`
        """
        results = [{} for _ in range(len(data))]
        # input check to early return
        if len(data) < 1 or self._schema_tree is None:
            return results

        # copy to stay `self._schema_tree` unchanged
        schema_list = self._schema_tree.children[:]
        while len(schema_list) > 0:
            node = schema_list.pop(0)
            examples = []
            input_map = {}
            cnt = 0
            idx = 0
            if not node.prefix:
                for one_data in data:
                    examples.append({'text': one_data, 'prompt': dbc2sbc(node.name)})
                    input_map[cnt] = [idx]
                    idx += 1
                    cnt += 1
            else:
                for pre, one_data in zip(node.prefix, data):
                    if len(pre) == 0:
                        input_map[cnt] = []
                    else:
                        for p in pre:
                            examples.append({'text': one_data, 'prompt': dbc2sbc(p + node.name)})
                        input_map[cnt] = [i + idx for i in range(len(pre))]
                        idx += len(pre)
                    cnt += 1
            if len(examples) == 0:
                result_list = []
            else:
                result_list = self._single_stage_predict(examples)

            if not node.parent_relations:
                relations = [[] for i in range(len(data))]
                for k, v in input_map.items():
                    for idx in v:
                        if len(result_list[idx]) == 0:
                            continue
                        if node.name not in results[k].keys():
                            results[k][node.name] = result_list[idx]
                        else:
                            results[k][node.name].extend(result_list[idx])
                    if node.name in results[k].keys():
                        relations[k].extend(results[k][node.name])
            else:
                relations = node.parent_relations
                for k, v in input_map.items():
                    for i in range(len(v)):
                        if len(result_list[v[i]]) == 0:
                            continue
                        if 'relations' not in relations[k][i].keys():
                            relations[k][i]['relations'] = {node.name: result_list[v[i]]}
                        elif node.name not in relations[k][i]['relations'].keys():
                            relations[k][i]['relations'][node.name] = result_list[v[i]]
                        else:
                            relations[k][i]['relations'][node.name].extend(result_list[v[i]])
                new_relations = [[] for i in range(len(data))]
                for i in range(len(relations)):
                    for j in range(len(relations[i])):
                        if 'relations' in relations[i][j].keys() and node.name in relations[i][j]['relations'].keys():
                            for k in range(len(relations[i][j]['relations'][node.name])):
                                new_relations[i].append(relations[i][j]['relations'][node.name][k])
                relations = new_relations

            prefix = [[] for _ in range(len(data))]
            for k, v in input_map.items():
                for idx in v:
                    for i in range(len(result_list[idx])):
                        if self.is_en:
                            prefix[k].append(' of ' + result_list[idx][i]['text'])
                        else:
                            prefix[k].append(result_list[idx][i]['text'] + '的')

            for child in node.children:
                child.prefix = prefix
                child.parent_relations = relations
                schema_list.append(child)
        return results

    @staticmethod
    def _convert_ids_to_results(examples, sentence_ids, probs):
        """
        Convert ids to raw text in a single stage.
        """
        results = []
        for example, sentence_id, prob in zip(examples, sentence_ids, probs):
            if len(sentence_id) == 0:
                results.append([])
                continue
            result_list = []
            text = example['text']
            prompt = example['prompt']
            for i in range(len(sentence_id)):
                start, end = sentence_id[i]
                if start < 0 and end >= 0:
                    continue
                if end < 0:
                    start += (len(prompt) + 1)
                    end += (len(prompt) + 1)
                    result = {'text': prompt[start:end], 'probability': prob[i]}
                    result_list.append(result)
                else:
                    result = {
                        'text': text[start:end],
                        'start': start,
                        'end': end,
                        'probability': prob[i]
                    }
                    result_list.append(result)
            results.append(result_list)
        return results

    @staticmethod
    def _auto_splitter(input_texts, max_text_len, split_sentence=False):
        """
        Split the raw texts automatically for model inference.
        Args:
            input_texts (List[str]): input raw texts.
            max_text_len (int): cutting length.
            split_sentence (bool): If True, sentence-level split will be performed.
        return:
            short_input_texts (List[str]): the short input texts for model inference.
            input_mapping (dict): mapping between raw text and short input texts.
        """
        input_mapping = {}
        short_input_texts = []
        cnt_org = 0
        cnt_short = 0
        for text in input_texts:
            if not split_sentence:
                sens = [text]
            else:
                sens = cut_chinese_sent(text)
            for sen in sens:
                lens = len(sen)
                if lens <= max_text_len:
                    short_input_texts.append(sen)
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = [cnt_short]
                    else:
                        input_mapping[cnt_org].append(cnt_short)
                    cnt_short += 1
                else:
                    temp_text_list = [
                        sen[i:i + max_text_len]
                        for i in range(0, lens, max_text_len)
                    ]
                    short_input_texts.extend(temp_text_list)
                    short_idx = cnt_short
                    cnt_short += math.ceil(lens / max_text_len)
                    temp_text_id = [
                        short_idx + i for i in range(cnt_short - short_idx)
                    ]
                    if cnt_org not in input_mapping.keys():
                        input_mapping[cnt_org] = temp_text_id
                    else:
                        input_mapping[cnt_org].extend(temp_text_id)
            cnt_org += 1
        return short_input_texts, input_mapping

    def _single_stage_predict(self, inputs):
        input_texts = []
        prompts = []
        for i in range(len(inputs)):
            input_texts.append(inputs[i]['text'])
            prompts.append(inputs[i]['prompt'])
        # max predict length should exclude the length of prompt and summary tokens
        max_predict_len = self.max_seq_len - len(max(prompts)) - 3
        short_input_texts, self.input_mapping = self._auto_splitter(input_texts, max_predict_len,
                                                                    split_sentence=self.split_sentence)

        short_texts_prompts = []
        for k, v in self.input_mapping.items():
            short_texts_prompts.extend([prompts[k] for i in range(len(v))])
        short_inputs = [
            {'text': short_input_texts[i], 'prompt': short_texts_prompts[i]} for i in range(len(short_input_texts))
        ]

        prompts = []
        texts = []
        for s in short_inputs:
            prompts.append(s['prompt'])
            texts.append(s['text'])
        if self.multilingual:
            padding_type = 'max_length'
        else:
            padding_type = 'longest'
        encoded_inputs = self.tokenizer(
            text=prompts,
            text_pair=texts,
            truncation=True,
            max_length=self.max_seq_len,
            padding=padding_type,
            add_special_tokens=True,
            return_tensors='np',
            return_offsets_mapping=True,
        )
        offset_maps = encoded_inputs['offset_mapping']

        start_probs = []
        end_probs = []
        for idx in range(0, len(texts), self.batch_size):
            l, r = idx, idx + self.batch_size
            input_ids = encoded_inputs['input_ids'][l:r]
            token_type_ids = encoded_inputs['token_type_ids'][l:r]
            attention_mask = encoded_inputs['attention_mask'][l:r]
            if self.multilingual:
                input_ids = np.array(input_ids, dtype="int64")
                attention_mask = np.array(attention_mask, dtype="int64")
                position_ids = (np.cumsum(np.ones_like(input_ids), axis=1) - np.ones_like(input_ids)) * attention_mask
                input_dict = {
                    'input_ids': input_ids.astype('int64'),
                    'position_ids': position_ids.astype('int64'),
                }
            else:
                input_dict = {
                    'input_ids': input_ids.astype('int64'),
                    'token_type_ids': token_type_ids.astype('int64'),
                    'attention_mask': attention_mask.astype('int64'),
                }
            start_prob, end_prob = self.inference_backend(input_dict)
            start_prob = start_prob.tolist()
            end_prob = end_prob.tolist()
            start_probs.extend(start_prob)
            end_probs.extend(end_prob)
        start_ids_list = get_bool_ids_greater_than(start_probs, limit=self.position_prob, return_prob=True)
        end_ids_list = get_bool_ids_greater_than(end_probs, limit=self.position_prob, return_prob=True)

        sentence_ids = []
        probs = []
        for start_ids, end_ids, offset_map in zip(start_ids_list, end_ids_list, offset_maps.tolist()):
            span_list = get_span(start_ids, end_ids, with_prob=True)
            sentence_id, prob = get_id_and_prob(span_list, offset_map)
            sentence_ids.append(sentence_id)
            probs.append(prob)

        results = self._convert_ids_to_results(short_inputs, sentence_ids, probs)
        results = self._auto_joiner(results, short_input_texts, self.input_mapping)
        return results

    @staticmethod
    def _auto_joiner(short_results, short_inputs, input_mapping):
        concat_results = []
        is_cls_task = False
        for short_result in short_results:
            if not short_result:
                continue
            elif 'start' not in short_result[0].keys() and 'end' not in short_result[0].keys():
                is_cls_task = True
                break
            else:
                break
        for k, vs in input_mapping.items():
            if is_cls_task:
                cls_options = {}
                for v in vs:
                    if len(short_results[v]) == 0:
                        continue
                    if short_results[v][0]['text'] not in cls_options.keys():
                        cls_options[short_results[v][0]['text']] = [1, short_results[v][0]['probability']]
                    else:
                        cls_options[short_results[v][0]['text']][0] += 1
                        cls_options[short_results[v][0]['text']][1] += short_results[v][0]['probability']
                if len(cls_options) != 0:
                    cls_res, cls_info = max(cls_options.items(), key=lambda x: x[1])
                    concat_results.append([{'text': cls_res, 'probability': cls_info[1] / cls_info[0]}])
                else:
                    concat_results.append([])
            else:
                offset = 0
                single_results = []
                for v in vs:
                    if v == 0:
                        single_results = short_results[v]
                        offset += len(short_inputs[v])
                    else:
                        for i in range(len(short_results[v])):
                            if 'start' not in short_results[v][
                                    i] or 'end' not in short_results[v][i]:
                                continue
                            short_results[v][i]['start'] += offset
                            short_results[v][i]['end'] += offset
                        offset += len(short_inputs[v])
                        single_results.extend(short_results[v])
                concat_results.append(single_results)
        return concat_results

    @classmethod
    def _build_tree(cls, schema, name='root'):
        """
        Build the schema tree.
        """
        schema_tree = SchemaTree(name)
        for s in schema:
            if isinstance(s, str):
                schema_tree.add_child(SchemaTree(s))
            elif isinstance(s, dict):
                for k, v in s.items():
                    if isinstance(v, str):
                        child = [v]
                    elif isinstance(v, list):
                        child = v
                    else:
                        raise TypeError(
                            'Invalid schema, value for each key:value pairs should be list or string'
                            'but {} received'.format(type(v)))
                    schema_tree.add_child(cls._build_tree(child, name=k))
            else:
                raise TypeError('Invalid schema, element should be string or dict, but {} received'.format(type(s)))
        return schema_tree

    def export_onnx(self):
        input_names = ['input_ids', 'token_type_ids', 'attention_mask']
        output_names = ['start_prob', 'end_prob']
        with torch.no_grad():
            model = self.model.to('cpu')
            model.eval()
            model.config.return_dict = True
            model.config.use_cache = False
            save_path = self.model_path + '/inference.onnx'
            dynamic_axes = {name: {0: 'batch', 1: 'sequence'} for name in chain(input_names, output_names)}
            # Generate dummy input
            batch_size = 2
            seq_length = 6
            dummy_input = [' '.join([self.tokenizer.unk_token]) * seq_length] * batch_size
            inputs = dict(self.tokenizer(dummy_input, return_tensors='pt'))
            torch.onnx.export(model, (inputs,), save_path, input_names=input_names, output_names=output_names,
                              dynamic_axes=dynamic_axes, do_constant_folding=True, opset_version=11)
        if not os.path.exists(save_path):
            self.logger.error(f'Export Failed!')
        self.logger.info(f'Covert onnx successful!')
        return save_path


class SchemaTree(object):
    """
    Implementation of SchemaTree
    """

    def __init__(self, name='root', children=None):
        self.name = name
        self.children = []
        self.prefix = None
        self.parent_relations = None
        if children is not None:
            for child in children:
                self.add_child(child)

    def __repr__(self):
        return self.name

    def add_child(self, node):
        assert isinstance(node, SchemaTree), 'The children of a node should be an instance of SchemaTree.'
        self.children.append(node)
