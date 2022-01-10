import cv2
import numpy as np
from torchvision.transforms import ToTensor
from lane import Lane
from configoox import Config
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


#cfg = Config("/home/LaneATT-main/cfgs/laneatt_culane_resnet34.yml")
cfg = Config("/home/LaneATT-main/cfgs/laneatt_ehl_wx_resnet34.yml")
model_path = "/home/LaneATT-main/experiments/laneatt_r34_ehlwx/models/model_0014.pt"

to_tensor = ToTensor()


def get_img(img_path):
    img = cv2.resize(cv2.imread(img_path), (640, 360))

    img = img / 255.
    # img = np.expand_dims(img, 0)
    img = to_tensor(img.astype(np.float32))
    return img


def get_epoch_model(model):
    return torch.load(model)['model']


model = cfg.get_model()
#model.load_state_dict(get_epoch_model(model_path))
####用ddp时保存模型用model.state_dict()(未用model.module.state_dict())   导致加载时缺失”key“
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path)['model'].items()})
model = model.to(torch.device('cuda'))
model.eval()
test_parameters = cfg.get_test_parameters()

img = torch.randn(1, 3, 360, 640)
img = img.cuda()
with torch.no_grad():
    output1, ot2 = model(img, **test_parameters)
    #prediction = model.decode(output, as_lanes=True)
    
    
# TorchScript export
try:
    print('\nStarting TorchScript export with torch %s...' % torch.__version__)
    f = model_path.replace('.pt', '.torchscript.pt')  # filename
    ts = torch.jit.trace(model, img)
    ts.save(f)
    print('TorchScript export success, saved as %s' % f)
except Exception as e:
    print('TorchScript export failure: %s' % e)



####ONNX export   --  目前转出的onnx存在问题，修改多数网络层未解决
"""
try:
    import onnx
    from onnxsim import simplify

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = model_path.replace('.pt', '.onnx')  # filename
    torch.onnx.export(model, img, f, verbose=True, opset_version=12, input_names=['images'],
                      output_names=['output'])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, f)
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)
"""