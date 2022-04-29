import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'out.onnx'
model_quant = 'outq.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant)
