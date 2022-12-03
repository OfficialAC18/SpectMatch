import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = '/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/model.onnx'
model_quant = '/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/model_quant_2.5s.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant,weight_type=QuantType.QUInt8)