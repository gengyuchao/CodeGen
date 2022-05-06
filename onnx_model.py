import os
import time

import numpy as np
from tqdm import tqdm
from onnxruntime import (
    GraphOptimizationLevel, InferenceSession,
    SessionOptions, get_all_providers
)
from transformers import GPT2TokenizerFast


def create_model_for_provider(
    model_path: str,
    provider: str = 'CPUExecutionProvider'
) -> InferenceSession:
    assert provider in get_all_providers(), \
        f"provider {provider} not found, {get_all_providers()}"
    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('NUM_THREADS', 4))
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session


# from jaxformer.hf.sample import create_custom_gpt2_tokenizer

def create_tokenizer(location='gpt2'):
    t = GPT2TokenizerFast.from_pretrained(location)
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer(location='gpt2'):
    t = create_tokenizer(location=location)
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


def predict(context, max_length=128, max_lines=10):
    tokenized = tokenizer(context)
    input_ids = tokenized['input_ids']
    outputs = ''
    pkv = np.zeros([20, 2, 1, 16, 1, 64]).astype(np.float32)
    # pkv = np.zeros([40, 16, 1, 64]).astype(np.float32)
    for _ in range(max_length):
        out, pkv = model.run(['output', 'pkv_output'], {
            "input": np.array([input_ids]).astype(np.int64),
            'pkv': pkv
        })
        token = out[:, -1, :].argmax()
        input_ids = [token]
        outputs += tokenizer.decode([token])
        if len(outputs.strip().split('\n')) >= max_lines:
            break
        if len(outputs.lstrip()) > 0 and outputs.endswith('\n\n'):
            break
    return outputs


start_time = time.time()
print('model loading')
CURERENT_DIR = os.path.realpath(os.path.dirname(__file__))
tokenizer_path = os.path.join(CURERENT_DIR, 'gpt2_tokenizer_fast')
if os.path.exists(tokenizer_path):
    tokenizer = create_custom_gpt2_tokenizer(tokenizer_path)
else:
    tokenizer = create_custom_gpt2_tokenizer()
model = create_model_for_provider(os.path.join(CURERENT_DIR, 'outq.onnx'))
print(f'model loaded {time.time() - start_time}')


if __name__ == '__main__':
    context = 'def avg(arr):'
    start = time.time()
    output = predict(context)
    print(time.time() - start)
    print(context + output)
