import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.algorithms.general_distill import GeneralDistill
from mmrazor.models import ALGORITHMS
from tkinter import _flatten
from .DeAug import Mulit_Augmentation
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
        self.collect_key = collect_key
        self.solve_number = solve_number
        self.convertor_training_epoch = convertor_training_epoch
        self.convertor_epoch_number = convertor_epoch_number
        self.convertor = Mulit_Augmentation(
            pretrain_path=pretrain_path,
            solve_number=self.solve_number
        )

        super(SDADistill, self).__init__(with_student_loss, with_teacher_loss, **kwargs)

    def train_step(self, data, optimizer):
        gt_bboxes = data["gt_bboxes"]
        img = data["img"]
        new_gt_bboxes, new_img = self.convertor(img,gt_bboxes)
        img = torch.cat([new_img,img],dim=0)
        data["gt_bboxes"] = gt_bboxes
        data["img"] = img
        return super(SDADistill, self).train_step(data, optimizer)
