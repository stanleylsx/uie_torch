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

import argparse
import collections
import json
import os
import pickle
import shutil
import numpy as np
import torch
from base64 import b64decode
try:
    import paddle
    from paddle.utils.download import get_path_from_url
    paddle_installed = True
except (ImportError, ModuleNotFoundError):
    from engines.utils.download_model import get_path_from_url
    paddle_installed = False
from engines.models.uie import UIE, UIEM

MODEL_MAP = {
    # vocab.txt/special_tokens_map.json/tokenizer_config.json are common to the default model.
    "uie-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json"
        }
    },
    "uie-medium": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medium/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-mini": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_mini/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-micro": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_micro/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-nano": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_nano/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-medical-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_medical_base_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base/tokenizer_config.json",
        }
    },
    "uie-base-en": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en_v1.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_base_en/tokenizer_config.json",
        }
    },
    # uie-m模型需要Ernie-M模型
    "uie-m-base": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/tokenizer_config.json",
            "sentencepiece.bpe.model":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/sentencepiece.bpe.model"

        }
    },
    "uie-m-large": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large_v1.0/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_large/tokenizer_config.json",
            "sentencepiece.bpe.model":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_m_base/sentencepiece.bpe.model"
        }
    },
    # Rename to `uie-medium` and the name of `uie-tiny` will be deprecated in future.
    "uie-tiny": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny_v0.1/model_state.pdparams",
            "model_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/model_config.json",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/vocab.txt",
            "special_tokens_map.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/special_tokens_map.json",
            "tokenizer_config.json":
            "https://bj.bcebos.com/paddlenlp/taskflow/information_extraction/uie_tiny/tokenizer_config.json"
        }
    },
    "ernie-3.0-base-zh": {
        "resource_file_urls": {
            "model_state.pdparams":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh.pdparams",
            "model_config.json":
            "base64:ew0KICAiYXR0ZW50aW9uX3Byb2JzX2Ryb3BvdXRfcHJvYiI6IDAuMSwNCiAgImhpZGRlbl9hY3QiOiAiZ2VsdSIsDQogICJoaWRkZW5fZHJvcG91dF9wcm9iIjogMC4xLA0KICAiaGlkZGVuX3NpemUiOiA3NjgsDQogICJpbml0aWFsaXplcl9yYW5nZSI6IDAuMDIsDQogICJtYXhfcG9zaXRpb25fZW1iZWRkaW5ncyI6IDIwNDgsDQogICJudW1fYXR0ZW50aW9uX2hlYWRzIjogMTIsDQogICJudW1faGlkZGVuX2xheWVycyI6IDEyLA0KICAidGFza190eXBlX3ZvY2FiX3NpemUiOiAzLA0KICAidHlwZV92b2NhYl9zaXplIjogNCwNCiAgInVzZV90YXNrX2lkIjogdHJ1ZSwNCiAgInZvY2FiX3NpemUiOiA0MDAwMCwNCiAgImluaXRfY2xhc3MiOiAiRXJuaWVNb2RlbCINCn0=",
            "vocab.txt":
            "https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_base_zh_vocab.txt",
            "special_tokens_map.json":
            "base64:eyJ1bmtfdG9rZW4iOiAiW1VOS10iLCAic2VwX3Rva2VuIjogIltTRVBdIiwgInBhZF90b2tlbiI6ICJbUEFEXSIsICJjbHNfdG9rZW4iOiAiW0NMU10iLCAibWFza190b2tlbiI6ICJbTUFTS10ifQ==",
            "tokenizer_config.json":
            "base64:eyJkb19sb3dlcl9jYXNlIjogdHJ1ZSwgInVua190b2tlbiI6ICJbVU5LXSIsICJzZXBfdG9rZW4iOiAiW1NFUF0iLCAicGFkX3Rva2VuIjogIltQQURdIiwgImNsc190b2tlbiI6ICJbQ0xTXSIsICJtYXNrX3Rva2VuIjogIltNQVNLXSIsICJ0b2tlbml6ZXJfY2xhc3MiOiAiRXJuaWVUb2tlbml6ZXIifQ=="
        }
    }
}


def build_params_map(model_prefix='encoder', attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's BERT
    :return:
    """
    weight_map = collections.OrderedDict({
        f'{model_prefix}.embeddings.word_embeddings.weight': "encoder.embeddings.word_embeddings.weight",
        f'{model_prefix}.embeddings.position_embeddings.weight': "encoder.embeddings.position_embeddings.weight",
        f'{model_prefix}.embeddings.token_type_embeddings.weight': "encoder.embeddings.token_type_embeddings.weight",
        f'{model_prefix}.embeddings.task_type_embeddings.weight': "encoder.embeddings.task_type_embeddings.weight",
        f'{model_prefix}.embeddings.layer_norm.weight': 'encoder.embeddings.LayerNorm.gamma',
        f'{model_prefix}.embeddings.layer_norm.bias': 'encoder.embeddings.LayerNorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.q_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.query.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.q_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.query.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.k_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.key.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.k_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.key.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.v_proj.weight'] = f'encoder.encoder.layer.{i}.attention.self.value.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.v_proj.bias'] = f'encoder.encoder.layer.{i}.attention.self.value.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.out_proj.weight'] = f'encoder.encoder.layer.{i}.attention.output.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.self_attn.out_proj.bias'] = f'encoder.encoder.layer.{i}.attention.output.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm1.weight'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.gamma'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm1.bias'] = f'encoder.encoder.layer.{i}.attention.output.LayerNorm.beta'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear1.weight'] = f'encoder.encoder.layer.{i}.intermediate.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear1.bias'] = f'encoder.encoder.layer.{i}.intermediate.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear2.weight'] = f'encoder.encoder.layer.{i}.output.dense.weight'
        weight_map[f'{model_prefix}.encoder.layers.{i}.linear2.bias'] = f'encoder.encoder.layer.{i}.output.dense.bias'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm2.weight'] = f'encoder.encoder.layer.{i}.output.LayerNorm.gamma'
        weight_map[f'{model_prefix}.encoder.layers.{i}.norm2.bias'] = f'encoder.encoder.layer.{i}.output.LayerNorm.beta'
    # add pooler
    weight_map.update(
        {
            f'{model_prefix}.pooler.dense.weight': 'encoder.pooler.dense.weight',
            f'{model_prefix}.pooler.dense.bias': 'encoder.pooler.dense.bias',
            'linear_start.weight': 'linear_start.weight',
            'linear_start.bias': 'linear_start.bias',
            'linear_end.weight': 'linear_end.weight',
            'linear_end.bias': 'linear_end.bias',
        }
    )
    return weight_map


def extract_and_convert(input_dir, output_dir, verbose=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if verbose:
        print('=' * 20 + 'save config file' + '=' * 20)
    config = json.load(open(os.path.join(input_dir, 'model_config.json'), 'rt', encoding='utf-8'))
    if 'init_args' in config:
        config = config['init_args'][0]
    config['architectures'] = ['UIE']
    config['layer_norm_eps'] = 1e-12
    del config['init_class']
    if 'sent_type_vocab_size' in config:
        config['type_vocab_size'] = config['sent_type_vocab_size']
    config['intermediate_size'] = 4 * config['hidden_size']
    json.dump(config, open(os.path.join(output_dir, 'config.json'),
              'wt', encoding='utf-8'), indent=4)
    if verbose:
        print('=' * 20 + 'save vocab file' + '=' * 20)
    shutil.copy(os.path.join(input_dir, 'vocab.txt'),
                os.path.join(output_dir, 'vocab.txt'))
    special_tokens_map = json.load(open(os.path.join(
        input_dir, 'special_tokens_map.json'), 'rt', encoding='utf-8'))
    json.dump(special_tokens_map, open(os.path.join(output_dir, 'special_tokens_map.json'),
              'wt', encoding='utf-8'))
    tokenizer_config = json.load(
        open(os.path.join(input_dir, 'tokenizer_config.json'), 'rt', encoding='utf-8'))
    if tokenizer_config['tokenizer_class'] == 'ErnieTokenizer':
        tokenizer_config['tokenizer_class'] = "BertTokenizer"
    json.dump(tokenizer_config, open(os.path.join(output_dir, 'tokenizer_config.json'),
              'wt', encoding='utf-8'))
    spm_file = os.path.join(input_dir, 'sentencepiece.bpe.model')
    if os.path.exists(spm_file):
        shutil.copy(spm_file, os.path.join(
            output_dir, 'sentencepiece.bpe.model'))
    if verbose:
        print('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = collections.OrderedDict()
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    weight_map.update(build_params_map('ernie', attention_num=config['num_hidden_layers']))
    if paddle_installed:
        import paddle.fluid.dygraph as D
        from paddle import fluid
        with fluid.dygraph.guard():
            paddle_paddle_params, _ = D.load_dygraph(
                os.path.join(input_dir, 'model_state'))
    else:
        paddle_paddle_params = pickle.load(
            open(os.path.join(input_dir, 'model_state.pdparams'), 'rb'))
        del paddle_paddle_params['StructuredToParameterName@@']
    for weight_name, weight_value in paddle_paddle_params.items():
        transposed = ''
        if 'weight' in weight_name:
            if '.encoder' in weight_name or 'pooler' in weight_name or 'linear' in weight_name:
                weight_value = weight_value.transpose()
                transposed = '.T'
        # Fix: embedding error
        if 'word_embeddings.weight' in weight_name:
            weight_value[0, :] = 0
        if weight_name not in weight_map:
            if verbose:
                print(f"{'='*20} [SKIP] {weight_name} {'='*20}")
            continue
        state_dict[weight_map[weight_name]] = torch.FloatTensor(weight_value)
        if verbose:
            print(f"{weight_name}{transposed} -> {weight_map[weight_name]} {weight_value.shape}")
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))


def check_model(input_model):
    if not os.path.exists(input_model):
        if input_model not in MODEL_MAP:
            raise ValueError('input_model not exists!')

        resource_file_urls = MODEL_MAP[input_model]['resource_file_urls']
        print('Downloading resource files...')

        for key, val in resource_file_urls.items():
            file_path = os.path.join(input_model, key)
            if not os.path.exists(file_path):
                if val.startswith('base64:'):
                    base64data = b64decode(val.replace('base64:', '').encode('utf-8'))
                    with open(file_path, 'wb') as f:
                        f.write(base64data)
                else:
                    download_path = get_path_from_url(val, input_model)
                    if download_path != file_path:
                        shutil.move(download_path, file_path)


def validate_model(tokenizer, pt_model, pd_model, model_type='uie', atol: float = 1e-5):

    print('Validating PyTorch model...')

    batch_size = 2
    seq_length = 6
    dummy_input = [" ".join([tokenizer.unk_token]) * seq_length] * batch_size
    encoded_inputs = dict(tokenizer(dummy_input, pad_to_max_seq_len=True, max_seq_len=512, return_attention_mask=True,
                                    return_position_ids=True))
    paddle_inputs = {}
    for name, value in encoded_inputs.items():
        if name == "attention_mask":
            if model_type == 'uie-m':
                continue
            name = "att_mask"
        if name == "position_ids":
            name = "pos_ids"
        paddle_inputs[name] = paddle.to_tensor(value, dtype=paddle.int64)

    paddle_named_outputs = ['start_prob', 'end_prob']
    paddle_outputs = pd_model(**paddle_inputs)

    torch_inputs = {}
    for name, value in encoded_inputs.items():
        if name == "attention_mask":
            if model_type == 'uie-m':
                continue
        torch_inputs[name] = torch.tensor(value, dtype=torch.int64)
    torch_outputs = pt_model(**torch_inputs)
    torch_outputs_dict = {}

    for name, value in torch_outputs.items():
        torch_outputs_dict[name] = value

    torch_outputs_set, ref_outputs_set = set(
        torch_outputs_dict.keys()), set(paddle_named_outputs)
    if not torch_outputs_set.issubset(ref_outputs_set):
        print(
            f"\t-[x] Pytorch model output names {torch_outputs_set} do not match reference model {ref_outputs_set}"
        )

        raise ValueError(
            "Outputs doesn't match between reference model and Pytorch converted model: "
            f"{torch_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        print(
            f"\t-[✓] Pytorch model output names match reference model ({torch_outputs_set})")

    # Check the shape and values match
    for name, ref_value in zip(paddle_named_outputs, paddle_outputs):
        ref_value = ref_value.numpy()
        pt_value = torch_outputs_dict[name].detach().numpy()
        print(f'\t- Validating PyTorch Model output "{name}":')

        # Shape
        if not pt_value.shape == ref_value.shape:
            print(
                f"\t\t-[x] shape {pt_value.shape} doesn't match {ref_value.shape}")
            raise ValueError(
                "Outputs shape doesn't match between reference model and Pytorch converted model: "
                f"Got {ref_value.shape} (reference) and {pt_value.shape} (PyTorch)"
            )
        else:
            print(
                f"\t\t-[✓] {pt_value.shape} matches {ref_value.shape}")

        # Values
        if not np.allclose(ref_value, pt_value, atol=atol):
            print(
                f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and Pytorch converted model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - pt_value))}"
            )
        else:
            print(
                f"\t\t-[✓] all values close (atol: {atol})")


def do_main():
    if args.output_model is None:
        args.output_model = args.input_model.replace('-', '_')+'_pytorch'
    check_model(args.input_model)
    extract_and_convert(args.input_model, args.output_model, verbose=True)
    if not (args.no_validate_output or 'ernie' in args.input_model):
        if paddle_installed:
            try:
                from paddlenlp.transformers import ErnieTokenizer, ErnieMTokenizer
                from paddlenlp.taskflow.models import UIE as UIEPaddle, UIEM as UIEMPaddle
            except (ImportError, ModuleNotFoundError) as e:
                raise ModuleNotFoundError(
                    'Module PaddleNLP is not installed. Try install paddlenlp or run convert.py with --no_validate_output') from e
            if 'uie-m' in args.input_model:
                tokenizer: ErnieMTokenizer = ErnieMTokenizer.from_pretrained(args.input_model)
                model = UIEM.from_pretrained(args.output_model)
                model.eval()
                paddle_model = UIEMPaddle.from_pretrained(args.input_model)
                paddle_model.eval()
                model_type = 'uie-m'
            else:
                tokenizer: ErnieTokenizer = ErnieTokenizer.from_pretrained(args.input_model)
                model = UIE.from_pretrained(args.output_model)
                model.eval()
                paddle_model = UIEPaddle.from_pretrained(args.input_model)
                paddle_model.eval()
                model_type = 'uie'
            validate_model(tokenizer, model, paddle_model, model_type)
        else:
            print('Skipping validating PyTorch model because paddle is not installed. '
                  'The outputs of the model may not be the same as Paddle model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_model", default="uie-base", type=str,
                        help="Directory of input paddle model.\n Will auto download model [uie-base/uie-tiny]")
    parser.add_argument("-o", "--output_model", default=None, type=str,
                        help="Directory of output pytorch model")
    parser.add_argument("--no_validate_output", action="store_true",
                        help="Directory of output pytorch model")
    args = parser.parse_args()

    do_main()
