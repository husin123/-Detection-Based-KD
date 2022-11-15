import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.algorithms.general_distill import GeneralDistill
from mmrazor.models import ALGORITHMS
from tkinter import _flatten
from .DeAug import Mulit_Augmentation
from .DV import detection_vis
@ALGORITHMS.register_module()
class SDADistill(GeneralDistill):
    def __init__(self,
                 collect_key,
                 solve_number,
                 convertor_training_epoch,
                 convertor_epoch_number,
                 pretrain_path,
                 with_student_loss=True,
                 with_teacher_loss=False,
                 **kwargs):
        super(SDADistill, self).__init__(with_student_loss, with_teacher_loss, **kwargs)
        self.collect_key = collect_key
        self.solve_number = solve_number
        self.convertor_training_epoch = convertor_training_epoch
        self.convertor_epoch_number = convertor_epoch_number
        self.convertor = Mulit_Augmentation(
            pretrain_path=pretrain_path,
            solve_number=self.solve_number
        )
    def train_step(self, data, optimizer):
        gt_bboxes = data["gt_bboxes"]
        gt_labels = data["gt_labels"]
        img = data["img"]
        new_img, new_gt_bboxes, new_gt_labels = self.convertor(img,gt_bboxes,gt_labels)
        img = torch.cat([new_img,img],dim=0)
        gt_bboxes = new_gt_bboxes + gt_bboxes
        gt_labels = new_gt_labels + gt_labels
        data["gt_bboxes"] = gt_bboxes
        data["gt_labels"] = gt_labels
        data["img"] = img
        data["img_metas"] = data["img_metas"] + copy.deepcopy(data["img_metas"])
        # PP = random.random()
        # detection_vis(img[0],gt_bboxes[0],gt_labels[0],"./image",f"0_{PP}")
        # detection_vis(img[2],gt_bboxes[2],gt_labels[2],"./image",f"2_{PP}")
        return super(SDADistill, self).train_step(data, optimizer)

from mmdet.datasets.pipelines.auto_augment import Rotate