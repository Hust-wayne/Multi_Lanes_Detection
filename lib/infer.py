import cv2
import numpy as np
from torchvision.transforms import ToTensor
from lane import Lane
from configoox import Config
import torch


cfg = Config("/home/LaneATT-main/cfgs/laneatt_tusimple_resnet34.yml")
model_path = "/home/LaneATT-main/experiments/laneatt_r34_tusimple/models/model_0009.pt"

to_tensor = ToTensor()



GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def get_img(img_path):
    img = cv2.resize(cv2.imread(img_path), (640, 360))

    img = img / 255.
    # img = np.expand_dims(img, 0)
    img = to_tensor(img.astype(np.float32))
    return img


def get_epoch_model(model):
    return torch.load(model)['model']


def draw_annotation(pred=None, img=None):

    #img = cv2.resize(img, (self.img_w, self.img_h))

    img_h, _, _ = img.shape
    # Pad image to visualize extrapolated predictions
    pad = 0
    if pad > 0:
        img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
        img_pad[pad:-pad, pad:-pad, :] = img
        img = img_pad
  
    for i, l in enumerate(pred):
        if l.metadata['type'] == 0:
            continue
 
        color = PRED_MISS_COLOR
        points = l.points
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        points = points.round().astype(int)
        points += pad
        xs, ys = points[:, 0], points[:, 1]
        for curr_p, next_p in zip(points[:-1], points[1:]):
            img = cv2.line(img,
                           tuple(curr_p),
                           tuple(next_p),
                           color=color,
                           thickness=3)
        if 'start_x' in l.metadata:
            start_x = l.metadata['start_x'] * img.shape[1]
            start_y = l.metadata['start_y'] * img.shape[0]
            cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                            radius=5,
                            color=(0, 0, 255),
                            thickness=-1)
        if len(xs) == 0:
            print("Empty pred")
        if len(xs) > 0:
            cv2.putText(img,
                     '{}-{:.0f}'.format(l.metadata['type'],l.metadata['conf'] * 100),
                     (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50)+(i+1)*20),
                     fontFace=cv2.FONT_HERSHEY_COMPLEX,
                     fontScale=0.7,
                    color=(255, 0, 255))
    return img
        


def eval(image_path):
    model = cfg.get_model()
    model.load_state_dict(get_epoch_model(model_path))
    model = model.to(torch.device('cuda'))
    model.eval()
    test_parameters = cfg.get_test_parameters()
    with torch.no_grad():
        image0 = get_img(image_path)
        image = image0.unsqueeze(0)
        image = image.to(torch.device('cuda'))
        output = model(image, **test_parameters)
        prediction = model.decode3(output, as_lanes=True)
        img = (image0.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = draw_annotation(img=img, pred=prediction[0])
        cv2.imwrite("nbv.jpg", img)
        #cv2.imshow('pred', img)
        #cv2.waitKey(0)
kk = '/home/LaneATT-main/data/figures/1.png'
eval(kk)