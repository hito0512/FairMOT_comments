import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict
import warnings
import cv2
import json
import numpy as np
import torch
import copy
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from opts import opts
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, generate_anchors, xywh2xyxy, encode_delta


class LoadImages:  # for inference
    def __init__(self, path, img_size=(640, 480)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(640, 480)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    
    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(640, 480), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, width=None, height=None):
        if height is None or width is None:
            height = self.height
            width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T, labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=480, width=640,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]

        return imw, targets, M
    else:
        return imw


# ---------- Predefined multi-scale input image width and height list
Input_WHs = [
    [256, 256],
    [320, 320],
    [416, 256],
    [448, 288],
    [480, 320],   # 0
    [512, 352],   # 1
    [544, 384],   # 2
    [576, 416],   # 3
    [608, 448],   # 4
    [640, 480],
    [640, 480],
    [640, 480]   # 15
]  # total 16 scales with floating aspect ratios


class MultiScaleJD(LoadImagesAndLabels):
    """
    multi-joint scale for trainig
    """
    mean = None
    std = None

    def __init__(self,
                 opt,
                 root,
                 paths,
                 img_size=(640, 480),
                 augment=False,
                 transforms=None):
        """
        :param opt:
        :param root:
        :param paths:
        :param img_size:
        :param augment:
        :param transforms:
        """
        self.opt = opt
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1  # C5: car, bicycle, person, cyclist, tricycle
        print("num_classes:", self.num_classes)  # 1
        # make sure img_size equal to opt.input_wh
        if opt.input_wh[0] != img_size[0] or opt.input_wh[1] != img_size[1]:
            opt.input_wh[0], opt.input_wh[1] = img_size[0], img_size[1]

        # default input width and height
        self.default_input_wh = opt.input_wh

        # net input width and height
        self.width = self.default_input_wh[0]
        self.height = self.default_input_wh[1]

        # define mapping from batch idx to scale idx
        self.batch_i_to_scale_i = defaultdict(int)

        # ----- generate img and label file path lists
        self.paths = paths
        for ds, path in self.paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [x.replace('images', 'labels_with_ids')
                                        .replace('.png', '.txt')
                                        .replace('.jpg', '.txt')
                                    for x in self.img_files[ds]]

            print('Total {} image files in {} dataset.'.format(len(self.label_files[ds]), ds))

        if opt.id_weight > 0:  # If do ReID calculation
            # @even: for MCMOT training
            for ds, label_paths in self.label_files.items():  # ??????????????????
                max_index = -1
                for lp in label_paths:
                    # print(lp)
                    lb = np.loadtxt(lp)
                    # print(lb)
                    if len(lb) < 1:
                        continue
                    if len(lb.shape) < 2:
                        img_max = lb[1]
                    else:
                        img_max = np.max(lb[:, 1])
                    if img_max > max_index:
                        max_index = img_max
                self.tid_num[ds] = max_index + 1
                # max_ids_dict = defaultdict(int)  # cls_id => max track id

                # # ?????????????????????label
                # for lp in label_paths:
                #     if not os.path.isfile(lp):
                #         print('[Warning]: invalid label file {}.'.format(lp))
                #         continue

                #     with warnings.catch_warnings():
                #         warnings.simplefilter("ignore")

                #         lb = np.loadtxt(lp)
                #         if len(lb) < 1:  # ???????????????
                #             continue

                #         lb = lb.reshape(-1, 6)
                #         for item in lb:  # label????????????item(????????????)
                #             if item[1] > max_ids_dict[int(item[0])]:  # item[0]: cls_id, item[1]: track id
                #                 max_ids_dict[int(item[0])] = item[1]

                # track id number
                # self.tid_num[ds] = max_ids_dict  # ??????????????????????????????reid???cls_id?????????dict

            # @even: for MCMOT training
            self.tid_start_idx_of_cls_ids = defaultdict(dict)
            last_idx_dict = defaultdict(int)  # ???0??????
            last_index = 0
            for k, v in self.tid_num.items():  # ???????????????????????????
                self.tid_start_index[k] = last_index
                last_index += v
                self.nID = int(last_index + 1)
                # for cls_id, id_num in v.items():  # ??????????????????????????????????????????, v?????????max_ids_dict
                #     self.tid_start_idx_of_cls_ids[k][cls_id] = last_idx_dict[cls_id]
                #     last_idx_dict[cls_id] += id_num

            # @even: for MCMOT training
            # self.nID_dict = defaultdict(int)
            # for k, v in last_idx_dict.items():
            #     self.nID_dict[k] = int(v)  # ???????????????tack ids??????

        self.nds = [len(x) for x in self.img_files.values()]  # ??????????????????(MOT15, MOT20...)????????????
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]  # ???????????????????????????????????????????
        self.nF = sum(self.nds)  # ????????????????????????????????????????????????
        self.max_objs = opt.K  # ??????????????????????????????????????????,??????128???
        self.augment = augment
        self.transforms = transforms

        print('dataset summary')
        print(self.tid_num)

        # if opt.id_weight > 0:  # If do ReID calculation
            # print('total # identities:', self.nID)
            # for k, v in self.nID_dict.items():
            #     print('Total {:d} IDs of {}'.format(v, id2cls[k]))

            # print('start index', self.tid_start_index)
            # for k, v in self.tid_start_idx_of_cls_ids.items():
            #     for cls_id, start_idx in v.items():
            #         print('Start index of dataset {} class {:d} is {:d}'
            #               .format(k, cls_id, start_idx))

        self.input_multi_scales = None

        if opt.gen_scale:  # whether to generate multi-scales while keeping aspect ratio
            self.gen_multi_scale_input_whs()

        # rand scale the first time
        self.rand_scale()
        print('Total {:d} multi-scales:\n'.format(len(self.input_multi_scales)), self.input_multi_scales)

    def rand_scale(self):
        # randomly generate mapping from batch idx to scale idx
        if self.input_multi_scales is None:
            self.num_batches = self.nF // self.opt.batch_size + 1
            for batch_i in range(self.num_batches):
                rand_batch_idx = np.random.randint(0, self.num_batches)
                rand_scale_idx = rand_batch_idx % len(Input_WHs)
                self.batch_i_to_scale_i[batch_i] = rand_scale_idx
        else:
            self.num_batches = self.nF // self.opt.batch_size + 1
            for batch_i in range(self.num_batches):
                rand_batch_idx = np.random.randint(0, self.num_batches)
                rand_scale_idx = rand_batch_idx % len(self.input_multi_scales)
                self.batch_i_to_scale_i[batch_i] = rand_scale_idx

    def gen_multi_scale_input_whs(self, num_scales=256, min_ratio=0.5, max_ratio=1.0):
        """
        generate input multi scale image sizes(w, h), keep default aspect ratio
        :param num_scales:
        :return:
        """
        gs = 32  # grid size

        self.input_multi_scales = [x for x in Input_WHs if not (x[0] % gs or x[1] % gs)]
        self.input_multi_scales.append([self.width, self.height])

        # ----- min scale and max scale
        # keep default aspect ratio
        self.default_aspect_ratio = self.height / self.width

        # min scale
        min_width = math.ceil(self.width * min_ratio / gs) * gs
        min_height = math.ceil(self.height * min_ratio / gs) * gs
        self.input_multi_scales.append([min_width, min_height])

        # max scale
        max_width = math.ceil(self.width * max_ratio / gs) * gs
        max_height = math.ceil(self.height * max_ratio / gs) * gs
        self.input_multi_scales.append([max_width, max_height])

        # other scales
        # widths = list(range(min_width, max_width + 1, int((max_width - min_width) / num_scales)))
        # heights = list(range(min_height, max_height + 1, int((max_height - min_height) / num_scales)))
        widths = list(range(min_width, max_width + 1, 1))
        heights = list(range(min_height, max_height + 1, 1))
        widths = [width for width in widths if not (width % gs)]
        heights = [height for height in heights if not (height % gs)]
        if len(widths) < len(heights):
            for width in widths:
                height = math.ceil(width * self.default_aspect_ratio / gs) * gs
                if [width, height] in self.input_multi_scales:
                    continue
                self.input_multi_scales.append([width, height])
        elif len(widths) > len(heights):
            for height in heights:
                width = math.ceil(height / self.default_aspect_ratio / gs) * gs
                if [width, height] in self.input_multi_scales:
                    continue
                self.input_multi_scales.append([width, height])
        else:
            for width, height in zip(widths, heights):
                if [width, height] in self.input_multi_scales:
                    continue
                height = math.ceil(width * self.default_aspect_ratio / gs) * gs
                self.input_multi_scales.append([width, height])

        if len(self.input_multi_scales) < 2:
            self.input_multi_scales = None
            print('[warning]: generate multi-scales failed(keeping aspect ratio)')
        else:
            self.input_multi_scales.sort(key=lambda x: x[0])

    def shuffle(self):
        """
        random shuffle the dataset
        :return:
        """
        tmp_img_files = copy.deepcopy(self.img_files)
        for ds, path in self.paths.items():
            ds_n_f = len(self.img_files[ds])  # number of files of this sub-dataset
            orig_img_files = self.img_files[ds]

            # re-generate ids
            uesd_ids = []
            for i in range(ds_n_f):
                new_idx = np.random.randint(0, ds_n_f)
                if new_idx in uesd_ids:
                    continue

                uesd_ids.append(new_idx)
                tmp_img_files[ds][i] = orig_img_files[new_idx]

        self.img_files = tmp_img_files  # re-evaluate img_files
        for ds, path in self.paths.items():  # re-evaluate corresponding label files
            self.label_files[ds] = [x.replace('images', 'labels_with_ids')
                                        .replace('.png', '.txt')
                                        .replace('.jpg', '.txt')
                                    for x in self.img_files[ds]]

    def __getitem__(self, idx):
        batch_i = idx // int(self.opt.batch_size)
        scale_idx = self.batch_i_to_scale_i[batch_i]
        if self.input_multi_scales is None:
            width, height = Input_WHs[scale_idx]
        else:
            width, height = self.input_multi_scales[scale_idx]

        # ???????????????????????????index
        for i, c in enumerate(self.cds):
            if idx >= c:
                ds = list(self.label_files.keys())[i]  # ??????idx?????????????????????
                start_index = c

        img_path = self.img_files[ds][idx - start_index]
        label_path = self.label_files[ds][idx - start_index]

        # Get image data and label: using multi-scale input image size
        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path, width, height)
        # print('input_h, input_w: %d %d' % (input_h, input_w))

        # ???????????????????????????, ????????????????????????(??????seq)?????????????????????index
        # @even: for MCMOT training
        if self.opt.id_weight > 0:
            for i, _ in enumerate(labels):
                if labels[i, 1] > -1:
                    labels[i, 1] += self.tid_start_index[ds]
                #     cls_id = int(labels[i][0])
                #     start_idx = self.tid_start_idx_of_cls_ids[ds][cls_id]
                #     labels[i, 1] += start_idx

        output_h = imgs.shape[1] // self.opt.down_ratio  # ??????????????????
        output_w = imgs.shape[2] // self.opt.down_ratio
        # print('output_h, output_w: %d %d' % (output_h, output_w))

        # ?????????????????????????????????
        # num_objs = labels.shape[0]
        # # --- GT of detection
        # hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)  # C??H??W: heat-map?????????????????????
        # wh = np.zeros((self.max_objs, opt.heads['wh']), dtype=np.float32)
        # reg = np.zeros((self.max_objs, opt.heads['reg']), dtype=np.float32)
        # ind = np.zeros((self.max_objs,), dtype=np.int64)  # K???object
        # reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)  # ?????????feature map?????????????????????reg loss
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        # if self.opt.id_weight > 0:
        #     # --- GT of ReID
        #     ids = np.zeros((self.max_objs,), dtype=np.int64)  # ????????????????????????ReID K?????????, ????????????id???0

        #     # @even: ?????????????????????????????????track ids
        #     cls_tr_ids = np.zeros((self.num_classes, output_h, output_w), dtype=np.int64)

        #     # @even, class id map: ??????(x, y)??????????????????, ???????????????-1
        #     cls_id_map = np.full((1, output_h, output_w), -1, dtype=np.int64)  # 1??H??W

        # Gauss function definition
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian

        # ???????????????ground truth????????????
        for k in range(num_objs):  # ??????????????????????????????
            label = labels[k]

            # ??????bbox????????????????????????GT???
            #                       0        1        2       3
            bbox = label[2:]  # center_x, center_y, bbox_w, bbox_h

            # ?????????????????????(?????????0??????, 0??????????????????)
            cls_id = int(label[0])

            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                # ?????? heat map
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                # ?????????feature????????????
                ind[k] = ct_int[1] * output_w + ct_int[0]
                # ???????????????
                reg[k] = ct - ct_int
                # mask??????
                reg_mask[k] = 1
                # ??????
                ids[k] = label[1]
                # bbox
                bbox_xys[k] = bbox_xy

        ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}
        return ret


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [640, 480]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(640, 480), augment=False, transforms=None):
        self.opt = opt
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]

        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                            bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1]
                bbox_xys[k] = bbox_xy

        ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}
        return ret


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(640, 480), augment=False, transforms=None):

        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs, labels0, img_path, (h, w)


