import logging

import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from scipy.interpolate import InterpolatedUnivariateSpline
from imgaug.augmentables.lines import LineString, LineStringsOnImage


from lib.lane import Lane

from .culane import CULane
from .tusimple import TuSimple
from .llamas import LLAMAS
from .nolabel_dataset import NoLabelDataset
from .ehl_wx import EHLWX
import pdb


GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class LaneDataset(Dataset):
    def __init__(self,
                 lane_num_types=None,
                 S=72,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=False,
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()
        self.use_dataset = dataset
        if dataset == 'tusimple':
            self.dataset = TuSimple(**kwargs)
        elif dataset == 'culane':
            self.dataset = CULane(**kwargs)
        elif dataset == 'llamas':
            self.dataset = LLAMAS(**kwargs)
        elif dataset == 'ehl_wx':
            self.dataset = EHLWX(**kwargs)
        elif dataset == 'nolabel_dataset':
            self.dataset = NoLabelDataset(**kwargs)
        else:
            raise NotImplementedError()
        self.lane_types = lane_num_types
        self.n_strips = S - 1
        self.n_offsets = S
        self.normalize = normalize
        self.img_h, self.img_w = img_size
        self.strip_size = self.img_h / self.n_strips
        self.logger = logging.getLogger(__name__)

        # y at each x offset
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size)
        self.transform_annotations()

        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation
        else:
            augmentations = []

        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_lanes = self.dataset.max_lanes

    @property
    def annotations(self):
        return self.dataset.annotations

    def transform_annotations(self):
        self.logger.info("Transforming annotations to the model's target format...")
        self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))
        self.logger.info('Done.')

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth()
            img_w = self.dataset.get_img_width()
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']

        old_lanes_type = anno['lane_type']

        # removing lanes with less than 2 points
        old_lanes_type = [old_lanes_type[ind] for ind,lane in enumerate(old_lanes) if len(lane) > 1]
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates
        old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
                     for lane in old_lanes]

        # create tranformed annotations
        # lanes = np.ones((self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
        #                 dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates

        ##### create multily type lane tranformed
        lanes = np.ones((self.dataset.max_lanes, self.lane_types + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 3 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates

        # lanes are invalid by default
        # lanes[:, 0] = 1
        # lanes[:, 1] = 0
        # #####多类别type须在这里增加维度
        # lanes[:, 2] = 0
        lane_type_onehot = np.eye(self.lane_types)
        lanes[:, :self.lane_types] = lane_type_onehot[0]

        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))   #水平堆叠

            ########多类别type须在这里增加 维度  类似 onehot
            # lanes[lane_idx, 0] = 0     # lane type scores
            # lanes[lane_idx, 1] = 0     # lane type
            # lanes[lane_idx, 2] = 1
            lanes[lane_idx, :self.lane_types] = lane_type_onehot[old_lanes_type[lane_idx]]

            lanes[lane_idx, self.lane_types] = len(xs_outside_image) / self.n_strips  # start_y
            lanes[lane_idx, self.lane_types+1] = xs_inside_image[0]                     # start_x
            lanes[lane_idx, self.lane_types+2] = len(xs_inside_image)                    # lane length
            lanes[lane_idx, (self.lane_types+3):(self.lane_types+3+len(all_xs))] = all_xs                   # all y 对应 all x points


            # lanes[lane_idx, 0] = 0     # lane type scores
            # lanes[lane_idx, 1] = 1     # lane type
            # lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips  # start_y
            # lanes[lane_idx, 3] = xs_inside_image[0]                     # start_x
            # lanes[lane_idx, 4] = len(xs_inside_image)                    # lane length
            # lanes[lane_idx, 5:5 + len(all_xs)] = all_xs                   # all y 对应 all x points
            
        #######multly lanetypes
        if self.use_dataset == 'culane':
            new_anno = {'path': os.path.join(self.dataset.root, anno['org_path']), 'label': lanes, 'lane_type': old_lanes_type, 'old_anno': anno}
        else:
            new_anno = {'path': anno['path'], 'label': lanes, 'lane_type': old_lanes_type, 'old_anno': anno}
        return new_anno
        
    def sample_lane(self, points, sample_ys):
        """

        :param points:   原始数据集中lane 点集
        :param sample_ys:  新定义的lane 点集中 y集合
        :return:
        """
        # this function expects the points to be sorted
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        assert len(points) > 1
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))  #将原始标注的lane进行拟合
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]   # 在原始数据label上转化为新定义的线段中 点 对应的 y 值
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain)   # 通过拟合 方程 通过 y 求出对应 x

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        #通过原始数据集中 lane 点集中 起始两点(自下而上) 向下延长 去拟合 新定义的点集  （可解决遮挡线问题）
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        all_xs = np.hstack((extrap_xs, interp_xs))

        # separate between inside and outside points
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w)   # 去掉超过 x 的边界点
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def draw_annotation(self, idx, label=None, pred=None, img=None):
        # Get image if not provided
        if img is None:
            img, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes3(label)
            img = img.permute(1, 2, 0).numpy()
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes3(label)
        img = cv2.resize(img, (self.img_w, self.img_h))

        img_h, _, _ = img.shape
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            assert len(matches) == len(pred)
            data.append((matches, accs, pred))
        else:
            fp = fn = None
        for matches, accs, datum in data:
            for i, l in enumerate(datum):
                if l.metadata['type'] == 0:
                    continue
                if matches is None:
                    color = GT_COLOR
                elif matches[i]:
                    color = PRED_HIT_COLOR
                else:
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
                                   thickness=3 if matches is None else 3)
                if 'start_x' in l.metadata:
                    start_x = l.metadata['start_x'] * img.shape[1]
                    start_y = l.metadata['start_y'] * img.shape[0]
                    cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                               radius=5,
                               color=(0, 0, 255),
                               thickness=-1)
                if len(xs) == 0:
                    print("Empty pred")
                if len(xs) > 0 and accs is not None:
                    cv2.putText(img,
                                '{}-{:.0f}'.format(l.metadata['type'], l.metadata['conf'] * 100),
                                (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50) + (i + 1) * 20),
                                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                fontScale=0.7,
                                color=(255, 0, 255))
        return img, fp, fn

    def label_to_lanes(self, label):
        lanes = []
        for l in label:
            if l[1] == 0:
                continue
            xs = l[6:] / self.img_w
            ys = self.offsets_ys / self.img_h
            start = int(round(l[3] * self.n_strips))
            length = int(round(l[5]))
            xs = xs[start:start + length][::-1]
            ys = ys[start:start + length][::-1]
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            points = np.hstack((xs, ys))

            lanes.append(Lane(points=points))
        return lanes

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img_org = cv2.imread(item['path'])
        
        self.use_dataset = None  #### 用于culane 数据增强
        
        try:
            line_strings_org = self.lane_to_linestrings(item['old_anno']['lanes'])
            line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape)
        except:
            print(item['path'])
        for i in range(30):
            img, line_strings = self.transform(image=img_org.copy(), line_strings=line_strings_org)
            line_strings.clip_out_of_image_()
            #######multly  label
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings), 'lane_type': item['old_anno']['lane_type']}
            try:
                label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()

        img = img / 255.
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.to_tensor(img.astype(np.float32))
        return (img, label, idx)

    def __len__(self):
        return len(self.dataset)

