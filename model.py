import os

from tqdm import tqdm
import torch
import torch.quantization.quantize_fx as quantize_fx

from jaxformer.hf.sample import truncate as do_truncate
from jaxformer.hf.sample import set_env, set_seed, print_time, create_model, create_custom_gpt2_tokenizer, create_tokenizer, sample

set_env()
models_pl = [
    'codegen-350M-multi', 'codegen-2B-multi', 'codegen-6B-multi',
    'codegen-16B-multi', 'codegen-350M-mono', 'codegen-2B-mono',
    'codegen-6B-mono', 'codegen-16B-mono'
]
use_cuda = not not os.environ.get('USE_CUDA')
fp16 = True if use_cuda else False


def load_model(chosen_model='codegen-350M-mono', pad=50256):
    """
    !wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-mono.tar.gz && tar -xvf checkpoints/codegen-350M-mono.tar.gz -C checkpoints/
    """
    ckpt = f'./checkpoints/{chosen_model}'
    if use_cuda:
        model = create_model(ckpt=ckpt, fp16=fp16).cuda().eval()
    else:
        model = create_model(ckpt=ckpt, fp16=fp16).to(torch.device('cpu')).eval()
        # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    tokenizer = create_custom_gpt2_tokenizer()
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = pad
    return tokenizer, model


def predict(tokenizer, model, context, p=0.95, t=0.2, max_length=128, batch_size=1, pad=50256):
    input_ids = tokenizer(
        context,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).input_ids

    completion = sample(
        model=model,
        tokenizer=tokenizer,
        context=context,
        pad_token_id=pad,
        num_return_sequences=batch_size,
        temp=t,
        top_p=p,
        max_length_sample=max_length,
        device=torch.device('cuda') if use_cuda else torch.device('cpu')
    )[0]
    truncation = do_truncate(completion)
    return truncation


with print_time('load model'):
    tokenizer, model = load_model()

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class MGPT(nn.Module):
    def __init__(self, model):
        super(MGPT, self).__init__()
        self.model = model

    def forward(self, input_ids = None, past_key_values = None):
        if past_key_values is not None:
            past_key_values = torch.unbind(past_key_values)
            past_key_values = [torch.unbind(x) for x in past_key_values]
        out = self.model(input_ids=input_ids, past_key_values=past_key_values)
        # [20, 2, 1, 16, x, 64]
        # dim 4 is free
        past_key_values = out.past_key_values
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])

        return out.logits, past_key_values


class MGPT_Multi(nn.Module):
    def __init__(self, model):
        super(MGPT_Multi, self).__init__()
        self.model = model

    def forward(self, input_ids = None, past_key_values = None):
        if past_key_values is not None:
            past_key_values = torch.unbind(past_key_values)
            past_key_values = [torch.unbind(x) for x in past_key_values]
        out = self.model(input_ids=input_ids, past_key_values=past_key_values)
        probs = F.softmax(out.logits[:, -1, :], dim=-1)
        next_inputs_ids = torch.multinomial(probs, 1)
        outputs = []
        outputs.append(next_inputs_ids)
        for i in range(63):
            probs = F.softmax(out.logits[:, -1, :], dim=-1)
            next_inputs_ids = torch.multinomial(probs, 1)
            # next_inputs_ids = out.logits[:, -1:, :].argmax(-1)
            outputs.append(next_inputs_ids)
            out = self.model(
                input_ids=next_inputs_ids,
                past_key_values=out.past_key_values
            )
        # [20, 2, 1, 16, x, 64]
        # dim 4 is free
        past_key_values = out.past_key_values
        past_key_values = torch.stack([torch.stack(x) for x in past_key_values])

        return torch.cat(outputs, dim=-1), past_key_values


def export_multi():
    """
    无法导出，会卡在导出命令上
    """
    global model
    os.environ['EXPORT'] = 'ok'
    context = '''def is_prime(n):'''
    tokenized = tokenizer(context)
    input_ids = torch.LongTensor([tokenized['input_ids']])
    all_ids = []
    # axis=4 is dynamic
    pkv = torch.zeros([20, 2, 1, 16, 1, 64])
    # num = torch.LongTensor([50])
    model = MGPT_Multi(model)

    # output, pkv = model(torch.LongTensor([input_ids]), pkv, torch.LongTensor([50]))
    # breakpoint()

    for i in tqdm(range(3)):
        output, pkv = model(input_ids, pkv)
        print(pkv.shape)
        tokens = output.detach().numpy()[0]
        all_ids += output.detach().numpy()[0].tolist()
        input_ids = output

    print(all_ids)
    print(tokenizer.decode(all_ids))

    print()
    print('start export')
    input_ids = torch.LongTensor([tokenized['input_ids']])
    pkv = torch.zeros([20, 2, 1, 16, 1, 64])
    num = torch.LongTensor([50])
    torch.onnx.export(
        model,               # model being run
        (input_ids, pkv),                         # model input (or a tuple for multiple inputs)
        "out_multi.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input', 'pkv'],   # the model's input names
        output_names = ['output', 'pkv_output'], # the model's output names
        dynamic_axes={
            'input' : {0 : 'batch_size', 1 : 'seq_len'},    # variable length axes
            'output' : {0 : 'batch_size', 1 : 'seq_len'},
            'pkv': {4: 'seq_len'},
            'pkv_output': {4: 'seq_len'},
        }
    )
    print('done')

def export():
    global model
    os.environ['EXPORT'] = 'ok'
    context = '''def is_prime(n):'''
    tokenized = tokenizer(context)
    input_ids = tokenized['input_ids']
    all_ids = []
    # axis=4 is dynamic
    pkv = torch.zeros([20, 2, 1, 16, 1, 64])
    model = MGPT(model)

    # output, pkv = model(torch.LongTensor([input_ids]), pkv, torch.LongTensor([50]))
    # breakpoint()

    for i in tqdm(range(20)):
        output, pkv = model(torch.LongTensor([input_ids]), pkv)
        print(pkv.shape)
        token = output.detach().numpy()[:, -1, :].argmax()
        all_ids += [token]
        input_ids = [token]

    print(all_ids)
    print(tokenizer.decode(all_ids))

    print()
    print('start export')
    input_ids = torch.LongTensor([input_ids])
    torch.onnx.export(
        model,               # model being run
        (input_ids, pkv),                         # model input (or a tuple for multiple inputs)
        "out.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=13,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input', 'pkv'],   # the model's input names
        output_names = ['output', 'pkv_output'], # the model's output names
        dynamic_axes={
            'input' : {0 : 'batch_size', 1 : 'seq_len'},    # variable length axes
            'output' : {0 : 'batch_size', 1 : 'seq_len'},
            'pkv': {4: 'seq_len'},
            'pkv_output': {4: 'seq_len'},
        }
    )
    print('done')


if __name__ == '__main__':
    export()
    # export_multi()
