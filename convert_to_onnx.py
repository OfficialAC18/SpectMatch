import torch.onnx
from torch import nn
from models.AST import build_AST
import onnxruntime as ort 
from onnxruntime_tools import optimizer
if __name__ == '__main__':
    device = 'cpu'

    model = build_AST(num_classes = 2 )
    #model = nn.DataParallel(model)
    model.load_state_dict(torch.load('/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/results/remasc_AST@334053/checkpoint.pt')['model_state_dict'])
    model.eval()
    # Use 'fbgemm' for server inference and 'qnnpack' for mobile inference
    backend = "fbgemm" # replaced with qnnpack causing much worse inference speed for quantized model on this notebook
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    #quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    print(model)
    #quantized_model.eval()

    x = torch.rand(1,250,128)
    
    torch.onnx.export(
        model,
        x,
        "model.onnx",
        input_names = ['input'],
        output_names = ['output'],
        export_params=True,
        do_constant_folding=True,
        dynamic_axes={
            'input': [0,1]
        }
    )
    '''
    model_onnx = ort.InferenceSession()
    onnxruntime.quantization.shape_inference(input_model_path = "/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/model.onnx",
                                             output_model_path = "/media/tower/d18329b1-fe35-43f8-8113-e096a40e5d3b/home/tower/Akchunya-Research/FixMatch (remasc) - PyTorch/model_opt.onnx",
                                             skip_onnx_shape = True)
'''