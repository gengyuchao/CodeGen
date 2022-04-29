import os
import math
import numpy as np
# import jieba
from onnxruntime import (
    GraphOptimizationLevel, InferenceSession,
    SessionOptions, get_all_providers
)
# from tokenization_enc_dec import EncDecTokenizer


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

from jaxformer.hf.sample import set_env, set_seed, print_time, create_model, create_custom_gpt2_tokenizer, create_tokenizer, sample
pad = 50256
tokenizer = create_custom_gpt2_tokenizer()
tokenizer.padding_side = 'left'
tokenizer.pad_token = pad

context = 'def avg():'
max_length = 512
input_ids = tokenizer(context, truncation=True, padding=True, max_length=max_length, return_tensors='np').input_ids

model = create_model_for_provider('out.onnx')
out = model.run(['output'], {"input": input_ids})[0]
breakpoint()
