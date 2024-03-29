import random
import math
import time
import os
import json
import numpy as np
from tqdm import tqdm
from decimal import Decimal
from config import configure


class DataConverter(object):
    """
    DataConverter to convert data export from annotation platform
    """

    def __init__(self, logger):
        """Init Data Converter"""
        self.logger = logger
        self.negative_ratio = configure['negative_ratio']
        self.prompt_prefix = '情感倾向'
        self.options = ['正向', '负向']
        self.separator = '##'
        self.schema_lang = 'ch'
        self.ignore_list = ['属性值', 'object']
        self.label_studio_file = configure['label_studio_file']
        self.save_dir = os.path.dirname(configure['train_file'])
        self.splits = configure['data_splits']
        self.task_type = configure['task_type']
        self.is_shuffle = True

    @staticmethod
    def process_text_tag(line, task_type='ext'):
        items = {'text': line['data']['text']}
        if task_type == 'ext':
            items['entities'] = []
            items['relations'] = []
            result_list = line['annotations'][0]['result']
            for a in result_list:
                if a['type'] == 'labels':
                    items['entities'].append(
                        {
                            'id': a["id"],
                            'start_offset': a['value']['start'],
                            'end_offset': a['value']['end'],
                            'label': a['value']['labels'][0],
                        }
                    )
                else:
                    items['relations'].append(
                        {
                            'id': a['from_id'] + '-' + a['to_id'],
                            'from_id': a['from_id'],
                            'to_id': a['to_id'],
                            'type': a['labels'][0],
                        }
                    )
        elif task_type == 'cls':
            items['label'] = line['annotations'][0]['result'][0]['value']['choices']
        return items

    def convert_cls_examples(self, raw_examples):
        """
        Convert labeled data for classification task.
        """
        examples = []
        self.logger.info('Converting annotation data...')
        with tqdm(total=len(raw_examples)):
            for line in raw_examples:
                items = self.process_text_tag(line, task_type='cls')
                text, labels = items['text'], items['label']
                example = self.generate_cls_example(text, labels, self.prompt_prefix)
                examples.append(example)
        return examples

    def convert_ext_examples(self, raw_examples, is_train=True):
        """
        Convert labeled data for extraction task.
        """

        def _sep_cls_label(label, separator):
            label_list = label.split(separator)
            if len(label_list) == 1:
                return label_list[0], None
            return label_list[0], label_list[1:]

        texts = []
        # {"content": "", "result_list": [], "prompt": "X"}
        entity_examples = []
        # {"content": "", "result_list": [], "prompt": "X的Y"}
        relation_examples = []
        # {"content": "", "result_list": [], "prompt": "X的情感倾向[正向，负向]"}
        entity_cls_examples = []

        # Entity label set: ["时间", "地点", ... ]
        entity_label_set = []
        # Entity name set: ["2月8日上午", "北京", ... ]
        entity_name_set = []
        # Predicate set: ["歌手", "所属专辑", ... ]
        predicate_set = []

        # List[List[str]]
        # List of entity prompt for each example
        entity_prompt_list = []
        # List of relation prompt for each example
        relation_prompt_list = []
        # Golden subject label for each example
        subject_golden_list = []
        # List of inverse relation for each example
        inverse_relation_list = []
        # List of predicate for each example
        predicate_list = []

        self.logger.info('Converting annotation data...')
        with tqdm(total=len(raw_examples)) as pbar:
            for line in raw_examples:
                items = self.process_text_tag(line, task_type='ext')
                text, relations, entities = items['text'], items['relations'], items['entities']
                texts.append(text)

                entity_example = []
                entity_prompt = []
                entity_example_map = {}
                entity_map = {}  # id to entity name
                for entity in entities:
                    entity_name = text[entity['start_offset']: entity['end_offset']]
                    entity_map[entity['id']] = {
                        'name': entity_name,
                        'start': entity['start_offset'],
                        'end': entity['end_offset'],
                    }
                    if entity['label'] in self.ignore_list:
                        continue

                    entity_label, entity_cls_label = _sep_cls_label(entity['label'], self.separator)

                    # Define the prompt prefix for entity-level classification
                    # xxx + "的" + 情感倾向 -> Chinese
                    # Sentiment classification + " of " + xxx -> English
                    if self.schema_lang == 'ch':
                        entity_cls_prompt_prefix = entity_name + '的' + self.prompt_prefix
                    else:
                        entity_cls_prompt_prefix = self.prompt_prefix + ' of ' + entity_name
                    if entity_cls_label is not None:
                        entity_cls_example = self.generate_cls_example(text, entity_cls_label, entity_cls_prompt_prefix)
                        entity_cls_examples.append(entity_cls_example)

                    result = {'text': entity_name, 'start': entity['start_offset'], 'end': entity['end_offset']}
                    if entity_label not in entity_example_map.keys():
                        entity_example_map[entity_label] = {
                            'content': text,
                            'result_list': [result],
                            'prompt': entity_label,
                        }
                    else:
                        entity_example_map[entity_label]['result_list'].append(result)

                    if entity_label not in entity_label_set and entity_label != '观点词':
                        entity_label_set.append(entity_label)
                    if entity_name not in entity_name_set:
                        entity_name_set.append(entity_name)
                    entity_prompt.append(entity_label)

                for v in entity_example_map.values():
                    entity_example.append(v)

                entity_examples.append(entity_example)
                entity_prompt_list.append(entity_prompt)

                subject_golden = []  # Golden entity inputs
                relation_example = []
                relation_prompt = []
                relation_example_map = {}
                inverse_relation = []
                predicates = []
                for relation in relations:
                    predicate = relation['type']
                    subject_id = relation['from_id']
                    object_id = relation['to_id']
                    # The relation prompt is constructed as follows:
                    # subject + "的" + predicate -> Chinese
                    # predicate + " of " + subject -> English
                    if self.schema_lang == 'ch':
                        prompt = entity_map[subject_id]['name'] + '的' + predicate
                        inverse_negative = entity_map[object_id]['name'] + '的' + predicate
                    else:
                        prompt = predicate + ' of ' + entity_map[subject_id]['name']
                        inverse_negative = predicate + ' of ' + entity_map[object_id]['name']

                    if entity_map[subject_id]['name'] not in subject_golden:
                        subject_golden.append(entity_map[subject_id]['name'])
                    result = {
                        'text': entity_map[object_id]['name'],
                        'start': entity_map[object_id]['start'],
                        'end': entity_map[object_id]['end'],
                    }

                    inverse_relation.append(inverse_negative)
                    predicates.append(predicate)

                    if prompt not in relation_example_map.keys():
                        relation_example_map[prompt] = {'content': text, 'result_list': [result], 'prompt': prompt}
                    else:
                        relation_example_map[prompt]['result_list'].append(result)

                    if predicate not in predicate_set:
                        predicate_set.append(predicate)
                    relation_prompt.append(prompt)

                for v in relation_example_map.values():
                    relation_example.append(v)

                relation_examples.append(relation_example)
                relation_prompt_list.append(relation_prompt)
                subject_golden_list.append(subject_golden)
                inverse_relation_list.append(inverse_relation)
                predicate_list.append(predicates)
                pbar.update(1)

        self.logger.info('Adding negative samples for first stage prompt...')
        positive_examples, negative_examples = self.add_entity_negative_example(
            entity_examples, texts, entity_prompt_list, entity_label_set)
        if len(positive_examples) == 0:
            all_entity_examples = []
        else:
            all_entity_examples = positive_examples + negative_examples

        all_relation_examples = []
        if len(predicate_set) != 0:
            self.logger.info('Adding negative samples for second stage prompt...')
            if is_train:

                positive_examples = []
                negative_examples = []
                per_n_ratio = self.negative_ratio // 3

                with tqdm(total=len(texts)) as pbar:
                    for i, text in enumerate(texts):
                        negative_example = []
                        collects = []
                        num_positive = len(relation_examples[i])

                        # 1. inverse_relation_list
                        redundants1 = inverse_relation_list[i]

                        # 2. entity_name_set ^ subject_golden_list[i]
                        redundants2 = []
                        if len(predicate_list[i]) != 0:
                            nonentity_list = list(set(entity_name_set) ^ set(subject_golden_list[i]))
                            nonentity_list.sort()

                            if self.schema_lang == 'ch':
                                redundants2 = [
                                    nonentity + '的' + predicate_list[i][random.randrange(len(predicate_list[i]))]
                                    for nonentity in nonentity_list
                                ]
                            else:
                                redundants2 = [
                                    predicate_list[i][random.randrange(len(predicate_list[i]))] + " of " + nonentity
                                    for nonentity in nonentity_list
                                ]

                        # 3. entity_label_set ^ entity_prompt_list[i]
                        redundants3 = []
                        if len(subject_golden_list[i]) != 0:
                            non_ent_label_list = list(set(entity_label_set) ^ set(entity_prompt_list[i]))
                            non_ent_label_list.sort()

                            if self.schema_lang == 'ch':
                                redundants3 = [
                                    subject_golden_list[i][random.randrange(len(subject_golden_list[i]))] + '的' + non_ent_label
                                    for non_ent_label in non_ent_label_list
                                ]
                            else:
                                redundants3 = [
                                    non_ent_label + ' of ' + subject_golden_list[i][random.randrange(len(subject_golden_list[i]))]
                                    for non_ent_label in non_ent_label_list
                                ]

                        redundants_list = [redundants1, redundants2, redundants3]

                        for redundants in redundants_list:
                            added, rest = self.add_relation_negative_example(
                                redundants,
                                texts[i],
                                num_positive,
                                per_n_ratio,
                            )
                            negative_example.extend(added)
                            collects.extend(rest)

                        num_sup = num_positive * self.negative_ratio - len(negative_example)
                        if num_sup > 0 and collects:
                            if num_sup > len(collects):
                                idxs = [k for k in range(len(collects))]
                            else:
                                idxs = random.sample(range(0, len(collects)), num_sup)
                            for idx in idxs:
                                negative_example.append(collects[idx])

                        positive_examples.extend(relation_examples[i])
                        negative_examples.extend(negative_example)
                        pbar.update(1)
                all_relation_examples = positive_examples + negative_examples
            else:
                relation_examples = self.add_full_negative_example(
                    relation_examples, texts, relation_prompt_list, predicate_set, subject_golden_list
                )
                all_relation_examples = [r for relation_example in relation_examples for r in relation_example]
        return all_entity_examples + all_relation_examples + entity_cls_examples

    def generate_cls_example(self, text, labels, prompt_prefix):
        random.shuffle(self.options)
        cls_options = ','.join(self.options)
        prompt = prompt_prefix + '[' + cls_options + ']'

        result_list = []
        example = {'content': text, 'result_list': result_list, 'prompt': prompt}
        for label in labels:
            start = prompt.rfind(label) - len(prompt) - 1
            end = start + len(label)
            result = {'text': label, 'start': start, 'end': end}
            example['result_list'].append(result)
        return example

    def add_full_negative_example(self, examples, texts, relation_prompt_list, predicate_set, subject_golden_list):
        with tqdm(total=len(relation_prompt_list)) as pbar:
            for i, relation_prompt in enumerate(relation_prompt_list):
                negative_sample = []
                for subject in subject_golden_list[i]:
                    for predicate in predicate_set:
                        # The relation prompt is constructed as follows:
                        # subject + "的" + predicate -> Chinese
                        # predicate + " of " + subject -> English
                        if self.schema_lang == 'ch':
                            prompt = subject + '的' + predicate
                        else:
                            prompt = predicate + ' of ' + subject
                        if prompt not in relation_prompt:
                            negative_result = {'content': texts[i], 'result_list': [], 'prompt': prompt}
                            negative_sample.append(negative_result)
                examples[i].extend(negative_sample)
                pbar.update(1)
        return examples

    def add_entity_negative_example(self, examples, texts, prompts, label_set):
        negative_examples = []
        positive_examples = []
        with tqdm(total=len(prompts)) as pbar:
            for i, prompt in enumerate(prompts):
                redundants = list(set(label_set) ^ set(prompt))
                redundants.sort()

                num_positive = len(examples[i])
                if num_positive != 0:
                    actual_ratio = math.ceil(len(redundants) / num_positive)
                else:
                    # Set num_positive to 1 for text without positive example
                    num_positive, actual_ratio = 1, 0

                if actual_ratio <= self.negative_ratio or self.negative_ratio == -1:
                    idxs = [k for k in range(len(redundants))]
                else:
                    idxs = random.sample(range(0, len(redundants)), self.negative_ratio * num_positive)

                for idx in idxs:
                    negative_result = {'content': texts[i], 'result_list': [], 'prompt': redundants[idx]}
                    negative_examples.append(negative_result)
                positive_examples.extend(examples[i])
                pbar.update(1)
        return positive_examples, negative_examples

    def add_relation_negative_example(self, redundants, text, num_positive, ratio):
        added_example = []
        rest_example = []

        if num_positive != 0:
            actual_ratio = math.ceil(len(redundants) / num_positive)
        else:
            # Set num_positive to 1 for text without positive example
            num_positive, actual_ratio = 1, 0

        all_idxs = [k for k in range(len(redundants))]
        if actual_ratio <= ratio or ratio == -1:
            idxs = all_idxs
            rest_idxs = []
        else:
            idxs = random.sample(range(0, len(redundants)), ratio * num_positive)
            rest_idxs = list(set(all_idxs) ^ set(idxs))

        for idx in idxs:
            negative_result = {'content': text, 'result_list': [], 'prompt': redundants[idx]}
            added_example.append(negative_result)

        for rest_idx in rest_idxs:
            negative_result = {'content': text, 'result_list': [], 'prompt': redundants[rest_idx]}
            rest_example.append(negative_result)

        return added_example, rest_example

    def do_convert(self):
        start_time = time.time()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if len(self.splits) != 0 and len(self.splits) != 3:
            raise ValueError('Only []/ len(splits)==3 accepted for splits.')

        def _check_sum():
            return Decimal(str(self.splits[0])) + Decimal(str(self.splits[1])) + \
                Decimal(str(self.splits[2])) == Decimal('1')

        if len(self.splits) == 3 and not _check_sum():
            raise ValueError('Please set correct splits, sum of elements in splits should be equal to 1.')

        with open(self.label_studio_file, 'r', encoding='utf-8') as f:
            raw_examples = json.loads(f.read())

        if self.is_shuffle:
            indexes = np.random.permutation(len(raw_examples))
            index_list = indexes.tolist()
            raw_examples = [raw_examples[i] for i in indexes]

        i1, i2, _ = self.splits
        p1 = int(len(raw_examples) * i1)
        p2 = int(len(raw_examples) * (i1 + i2))

        train_ids = index_list[:p1]
        dev_ids = index_list[p1:p2]
        test_ids = index_list[p2:]

        with open(os.path.join(self.save_dir, 'sample_index.json'), 'w') as fp:
            maps = {'train_ids': train_ids, 'dev_ids': dev_ids, 'test_ids': test_ids}
            fp.write(json.dumps(maps))

        if self.task_type == 'ext':
            train_examples = self.convert_ext_examples(raw_examples[:p1])
            dev_examples = self.convert_ext_examples(raw_examples[p1:p2], is_train=False)
            test_examples = self.convert_ext_examples(raw_examples[p2:], is_train=False)
        else:
            train_examples = self.convert_cls_examples(raw_examples[:p1])
            dev_examples = self.convert_cls_examples(raw_examples[p1:p2])
            test_examples = self.convert_cls_examples(raw_examples[p2:])

        def _save_examples(file_name, examples):
            count = 0
            save_path = os.path.join(self.save_dir, file_name)
            with open(save_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    count += 1
            self.logger.info('Save %d examples to %s.' % (count, save_path))

        _save_examples('train.txt', train_examples)
        _save_examples('dev.txt', dev_examples)
        _save_examples('test.txt', test_examples)

        self.logger.info('Finished! It takes %.2f seconds' % (time.time() - start_time))
