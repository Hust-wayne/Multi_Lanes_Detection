import cv2
import numpy as np
from torchvision.transforms import ToTensor
from lane import Lane
import torch
from nms import nms
import torch.nn as nn
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

model_path = "/home/LaneATT-main/experiments/laneatt_r34_ehlwx/models/model_0014.torchscript.pt"

to_tensor = ToTensor()

GT_COLOR = {"1":(255, 0, 0), "2":(55, 28, 206),"3":(35, 198, 26), "4":(35, 78, 210), 
            "5":(255, 255, 0), "6":(55, 255, 206),"7":(255, 18, 26), "8":(255, 78, 210),
            "9":(255, 0, 255), "10":(55, 28, 255),"11":(35, 198, 255), "12":(35, 78, 255)}
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
img_w, img_h = 640, 360
num_lane_types = 13   # bg + 12
n_strips = 71
anchor_ys = torch.linspace(1, 0, steps=72, dtype=torch.float32)


def draw_annotation(pred=None, img=None):
    pad = 0
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
        
        if int(l.metadata['type']) in [1,3,5,7,9]:   ##虚线
            for curr_p in points:
                img = cv2.circle(img, tuple(curr_p), 5, GT_COLOR[str(int(l.metadata['type']))], -1)
        else:
            for curr_p, next_p in zip(points[:-1], points[1:]):
                img = cv2.line(img,
                               tuple(curr_p),
                               tuple(next_p),
                               color=GT_COLOR[str(int(l.metadata['type']))],
                               thickness=3)
        """
        if 'start_x' in l.metadata:
            start_x = l.metadata['start_x'] * img.shape[1]
            start_y = l.metadata['start_y'] * img.shape[0]
            cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                       radius=5,
                       color=(0, 0, 255),
                       thickness=-1)
        """
        if len(xs) == 0:
            print("Empty pred")
        if len(xs) > 0:
            cv2.putText(img,
                        '{}-{:.0f}'.format(l.metadata['type'], l.metadata['conf'] * 100),
                        (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50) + (i + 1) * 20),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX,
                        fontScale=0.7,
                        color=(255, 0, 255))
    return img


def nms_func(proposals, scores, nms_thres, nms_topk, conf_threshold=0.2):
    if proposals.shape[0] == 0:
        return None
    above_threshold = scores > conf_threshold
    proposals = proposals[above_threshold]
    scores = scores[above_threshold]
    keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
    keep = keep[:num_to_keep]
    proposals = proposals[keep]
    return proposals


def proposals_to_pred(proposals):
    global anchor_ys
    anchor_ys = anchor_ys.to(proposals.device)
    anchor_ys = anchor_ys.double()
    lanes = []
    for lane in proposals:
        lane_xs = lane[(num_lane_types + 3):] / img_w
        start = int(round(lane[num_lane_types].item() * n_strips))
        length = int(round(lane[num_lane_types + 2].item()))
        end = start + length - 1
        end = min(end, len(anchor_ys) - 1)
        # end = label_end
        # if the proposal does not start at the bottom of the image,
        # extend its proposal until the x is outside the image
        mask = ~((((lane_xs[:start] >= 0.) &
                   (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool))
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2
        lane_ys = anchor_ys[lane_xs >= 0]
        lane_xs = lane_xs[lane_xs >= 0]
        lane_xs = lane_xs.flip(0).double()
        lane_ys = lane_ys.flip(0)
        if len(lane_xs) <= 1:
            continue
        points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
        lane = Lane(points=points.cpu().numpy(),
                    metadata={
                        'start_x': lane[num_lane_types + 1]/img_w,
                        'start_y': lane[num_lane_types]/img_h,
                        'conf': lane[:num_lane_types].max(),
                        'type': lane[:num_lane_types].argmax()
                    })
        lanes.append(lane)
    return lanes


def decode(proposals):
    if proposals.shape[0] == 0:
        return None
    softmax = nn.Softmax(dim=1)
    proposals[:, :num_lane_types] = softmax(proposals[:, :num_lane_types])
    proposals[:, num_lane_types + 2] = torch.round(proposals[:, num_lane_types + 2])
    pred = proposals_to_pred(proposals)
    return pred
    

def create_video(filename, size, fps=5):
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter(filename, fourcc, float(fps), size)

    return video


def eval_video(video_path):
    model = torch.jit.load(model_path)
    videoCapture = cv2.VideoCapture(video_path)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video = create_video(os.path.join(os.path.dirname(video_path), "out34.avi"), size=size, fps=15)
    with torch.no_grad():
        success, img_data = videoCapture.read()
        st = time.time()
        while success:
            pst = time.time()
            image0 = img_data
            image = cv2.resize(image0, (640, 360))
            image = image / 255.
            image = to_tensor(image.astype(np.float32))
            image = image.unsqueeze(0)
            image = image.to(torch.device('cuda'))
            proposal, scores = model(image)
            result = nms_func(proposal, scores, 50, 4, 0.5)
            prediction = decode(result)
            print(time.time() -pst)
            img = image0
            if prediction:
                img = draw_annotation(img=img, pred=prediction)
            else:
                print("NO  NO NO")
            video.write(img)

            success, img_data = videoCapture.read()
        print(time.time() -st)
            

def eval(image_path, save_path):
    model = torch.jit.load(model_path)
    #video = create_video("wxout.avi", size=(1640,590), fps=25)   #culane
    #video = create_video("out.avi", size=(1276,717), fps=5)   #LLAMAS
    with torch.no_grad():
        i = 0
        for file in os.listdir(image_path):
            i+=1
            st = time.time()
            file_path = os.path.join(image_path, file)
            image0 = cv2.imread(file_path)
            image = cv2.resize(image0, (640, 360))
            image = image / 255.
            image = to_tensor(image.astype(np.float32))
            image = image.unsqueeze(0)
            image = image.to(torch.device('cuda'))
            
            proposal, scores = model(image)
            result = nms_func(proposal, scores, 50, 4, 0.2)
           
            prediction = decode(result)
            print(time.time() -st)
            #img = (image.cpu().permute(2, 1, 0).numpy() * 255).astype(np.uint8)
            img = image0
            if prediction:
                img = draw_annotation(img=img, pred=prediction)
            cv2.imwrite(os.path.join(save_path,str(i) + "nbv.jpg"), img)
            #video.write(img)
            
            

#kk = '/home/LaneATT-main/data/video_example'
#kk = "/notebooks/LLAMAS/train/images-2014-12-22-14-01-36_mapping_280N_3rd_lane"
#kk = '/home/LaneATT-main/data/images2'
#save = '/home/LaneATT-main/data/result'
#eval(kk,save)

vd = '/home/LaneATT-main/data/fig/34.mp4'
eval_video(vd)
