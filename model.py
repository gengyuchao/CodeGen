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

import numpy as np

def export():
    os.environ['EXPORT'] = 'ok'
    context = '''def is_prime(n):'''
    tokenized = tokenizer(context)
    input_ids = tokenized['input_ids']
    all_ids = []
    # axis=2 is dynamic
    pkv = torch.zeros([40, 16, 1, 64])

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
            'pkv': {2: 'seq_len'},
            'pkv_output': {2: 'seq_len'},
        }
    )
    print('done')


if __name__ == '__main__':
    export()
