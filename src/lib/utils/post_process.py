from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  # 对于每个batch
  for i in range(dets.shape[0]):
    top_preds = {}
    # 输入dets的左上角，中心点，最长边，(128， 128):heatmap的长宽
    # 将bbox的左上角和右下角的坐标映射到原图上,求出坐标大小
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    # 将bbox的坐标和类别信息都放到字典top_preds中, top_pred有80类,
    # 每个类别中是一个list, list中每个元素包含了每个bbox的坐标和类别
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    #对batch里的每张图片都有一个top_preds字典来存储输出信息
    ret.append(top_preds)
  return ret

