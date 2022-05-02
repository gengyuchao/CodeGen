import os
import math

import numpy as np
from tqdm import tqdm
from onnxruntime import (
    GraphOptimizationLevel, InferenceSession,
    SessionOptions, get_all_providers
)
from transformers import GPT2TokenizerFast

CURERENT_DIR = os.path.realpath(os.path.dirname(__file__))


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

def create_tokenizer():
    t = GPT2TokenizerFast.from_pretrained('gpt2')
    t.max_model_input_sizes['gpt2'] = 1e20
    return t


def include_whitespace(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens([' ' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def include_tabs(t, n_min=2, n_max=20, as_special_tokens=False):
    t.add_tokens(['\t' * n for n in reversed(range(n_min, n_max))], special_tokens=as_special_tokens)
    return t


def create_custom_gpt2_tokenizer():
    t = create_tokenizer()
    t = include_whitespace(t=t, n_min=2, n_max=32, as_special_tokens=False)
    t = include_tabs(t=t, n_min=2, n_max=10, as_special_tokens=False)
    return t


tokenizer = create_custom_gpt2_tokenizer()
model = create_model_for_provider('outq.onnx')


def predict(context, max_length=512, max_lines=3):
    tokenized = tokenizer(context)
    input_ids = tokenized['input_ids']
    outputs = ''
    pkv = np.zeros([40, 16, 1, 64]).astype(np.float32)
    for i in range(max_length):
        out, pkv = model.run(['output', 'pkv_output'], {
            "input": np.array([input_ids]),
            'pkv': pkv
        })
        token = out[:, -1, :].argmax()
        input_ids = [token]
        outputs += tokenizer.decode([token])
        if len(outputs.strip().split('\n')) >= max_lines:
            break
        if len(outputs.lstrip()) > 0 and outputs.endswith('\n'):
            break
    return outputs


if __name__ == '__main__':
    import time
    context = 'def avg(arr):'
    start = time.time()
    output = predict(context)
    print(time.time() - start)
    print(context + output)
