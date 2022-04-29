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

def load_model(chosen_model='codegen-350M-mono',
               fp16=False,
               pad=50256):
    """
    !wget -P checkpoints https://storage.googleapis.com/sfr-codegen-research/checkpoints/codegen-350M-mono.tar.gz && tar -xvf checkpoints/codegen-350M-mono.tar.gz -C checkpoints/
    """
    ckpt = f'./checkpoints/{chosen_model}'
    model = create_model(ckpt=ckpt, fp16=fp16).to(torch.device('cpu')).eval()
    # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear},
    #                                             dtype=torch.qint8)
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

    completion = sample(model=model,
                        tokenizer=tokenizer,
                        context=context,
                        pad_token_id=pad,
                        num_return_sequences=batch_size,
                        temp=t,
                        top_p=p,
                        max_length_sample=max_length,
                        device=torch.device('cpu'))[0]
    truncation = do_truncate(completion)
    return context + truncation


with print_time('load model'):
    tokenizer, model = load_model()

import numpy as np

def export():
    context = 'print the code'
    max_length = 512
    input_ids = tokenizer(
        context,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt',
    ).input_ids
    
    # input_ids = np.array(input_ids)
    # breakpoint()

    torch.onnx.export(
        model,               # model being run
        (input_ids, ),                         # model input (or a tuple for multiple inputs)
        "out.onnx",   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={
            'input' : {0 : 'batch_size', 1 : 'seq_len'},    # variable length axes
            'output' : {0 : 'batch_size', 1 : 'seq_len'}
        }
    )


if __name__ == '__main__':
    # torch.save(model.state_dict(), 'PATH')
    # tokenizer.save_pretrained("local-pt-checkpoint")
    # model.save_pretrained("local-pt-checkpoint")
    # onnx_config = model_onnx_config(model.config)
    # from transformers.onnx.convert import export_pytorch
    # export_pytorch(tokenizer, model)
    export()
