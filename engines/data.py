from torch.utils.data import Dataset
import numpy as np
import json
import re


def reader(data_path, max_seq_len=512):
    """
    read json
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_line = json.loads(line)
            content = json_line['content']
            prompt = json_line['prompt']
            # Model Input is aslike: [CLS] Prompt [SEP] Content [SEP]
            # It include three summary tokens.
            if max_seq_len <= len(prompt) + 3:
                raise ValueError('The value of max_seq_len is too small, please set a larger value')
            max_content_len = max_seq_len - len(prompt) - 3
            if len(content) <= max_content_len:
                yield json_line
            else:
                result_list = json_line['result_list']
                json_lines = []
                accumulate = 0
                while True:
                    cur_result_list = []

                    for result in result_list:
                        if result['start'] + 1 <= max_content_len < result['end']:
                            max_content_len = result['start']
                            break

                    cur_content = content[:max_content_len]
                    res_content = content[max_content_len:]

                    while True:
                        if len(result_list) == 0:
                            break
                        elif result_list[0]['end'] <= max_content_len:
                            if result_list[0]['end'] > 0:
                                cur_result = result_list.pop(0)
                                cur_result_list.append(cur_result)
                            else:
                                cur_result_list = [result for result in result_list]
                                break
                        else:
                            break

                    json_line = {'content': cur_content, 'result_list': cur_result_list, 'prompt': prompt}
                    json_lines.append(json_line)

                    for result in result_list:
                        if result['end'] <= 0:
                            break
                        result['start'] -= max_content_len
                        result['end'] -= max_content_len
                    accumulate += max_content_len
                    max_content_len = max_seq_len - len(prompt) - 3
                    if len(res_content) == 0:
                        break
                    elif len(res_content) < max_content_len:
                        json_line = {'content': res_content, 'result_list': result_list, 'prompt': prompt}
                        json_lines.append(json_line)
                        break
                    else:
                        content = res_content

                for json_line in json_lines:
                    yield json_line


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def convert_example(example, tokenizer, max_seq_len):
    """
    example: {
        title
        prompt
        content
        result_list
    }
    """
    encoded_inputs = tokenizer(
        text=[example['prompt']],
        text_pair=[example['content']],
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=True,
        return_offsets_mapping=True)
    offset_mapping = [list(x) for x in encoded_inputs['offset_mapping'][0]]
    bias = 0
    for index in range(1, len(offset_mapping)):
        mapping = offset_mapping[index]
        if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
            bias = offset_mapping[index - 1][1] + 1  # Includes [SEP] token
        if mapping[0] == 0 and mapping[1] == 0:
            continue
        offset_mapping[index][0] += bias
        offset_mapping[index][1] += bias
    start_ids = [0 for x in range(max_seq_len)]
    end_ids = [0 for x in range(max_seq_len)]
    for item in example['result_list']:
        start = map_offset(item['start'] + bias, offset_mapping)
        end = map_offset(item['end'] - 1 + bias, offset_mapping)
        start_ids[start] = 1.0
        end_ids[end] = 1.0

    tokenized_output = [
        encoded_inputs['input_ids'][0], encoded_inputs['token_type_ids'][0], encoded_inputs['attention_mask'][0],
        start_ids, end_ids
    ]
    tokenized_output = [np.array(x, dtype='int64') for x in tokenized_output]
    tokenized_output = [np.pad(x, (0, max_seq_len-x.shape[-1]), 'constant') for x in tokenized_output]
    return tuple(tokenized_output)


def unify_prompt_name(prompt):
    # The classification labels are shuffled during finetuning, so they need
    # to be unified during evaluation.
    if re.search(r'\[.*?\]$', prompt):
        prompt_prefix = prompt[:prompt.find("[", 1)]
        cls_options = re.search(r'\[.*?\]$', prompt).group()[1:-1].split(",")
        cls_options = sorted(list(set(cls_options)))
        cls_options = ",".join(cls_options)
        prompt = prompt_prefix + "[" + cls_options + "]"
        return prompt
    return prompt


class IEDataset(Dataset):
    """
    Dataset for Information Extraction fron jsonl file.
    The line type is
    {
        content
        result_list
        prompt
    }
    """

    def __init__(self, file_path, tokenizer, max_seq_len) -> None:
        super().__init__()
        self.file_path = file_path
        self.dataset = list(reader(file_path))
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return convert_example(self.dataset[index], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)


def get_relation_type_dict(relation_data):

    def compare(a, b):
        a = a[::-1]
        b = b[::-1]
        res = ''
        for i in range(min(len(a), len(b))):
            if a[i] == b[i]:
                res += a[i]
            else:
                break
        if res == "":
            return res
        elif res[::-1][0] == "的":
            return res[::-1][1:]
        return ""

    relation_type_dict = {}
    added_list = []
    for i in range(len(relation_data)):
        added = False
        if relation_data[i][0] not in added_list:
            for j in range(i + 1, len(relation_data)):
                match = compare(relation_data[i][0], relation_data[j][0])
                if match != "":
                    match = unify_prompt_name(match)
                    if relation_data[i][0] not in added_list:
                        added_list.append(relation_data[i][0])
                        relation_type_dict.setdefault(match, []).append(
                            relation_data[i][1])
                    added_list.append(relation_data[j][0])
                    relation_type_dict.setdefault(match, []).append(
                        relation_data[j][1])
                    added = True
            if not added:
                added_list.append(relation_data[i][0])
                suffix = relation_data[i][0].rsplit("的", 1)[1]
                suffix = unify_prompt_name(suffix)
                relation_type_dict[suffix] = relation_data[i][1]
    return relation_type_dict


class IEMapDataset(Dataset):
    """
    Dataset for Information Extraction from jsonl file.
    The line type is
    {
        content
        result_list
        prompt
    }
    """

    def __init__(self, data, tokenizer, max_seq_len) -> None:
        super().__init__()
        self.dataset = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return convert_example(self.dataset[index], tokenizer=self.tokenizer, max_seq_len=self.max_seq_len)


def get_id_and_prob(spans, offset_map):
    prompt_length = 0
    for i in range(1, len(offset_map)):
        if offset_map[i] != [0, 0]:
            prompt_length += 1
        else:
            break

    for i in range(1, prompt_length + 1):
        offset_map[i][0] -= (prompt_length + 1)
        offset_map[i][1] -= (prompt_length + 1)

    sentence_id = []
    prob = []
    for start, end in spans:
        prob.append(start[1] * end[1])
        sentence_id.append(
            (offset_map[start[0]][0], offset_map[end[0]][1]))
    return sentence_id, prob

