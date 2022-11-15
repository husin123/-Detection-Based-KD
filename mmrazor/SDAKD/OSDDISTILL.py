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
from torch.nn.parallel import DistributedDataParallel as DDP
import os,copy
from mmrazor.models.utils import add_prefix
from mmcv.runner.hooks import OptimizerHook

@ALGORITHMS.register_module()
class OSDDistill(GeneralDistill):
    def __init__(self,
                 collect_key,
                 solve_number,
                 convertor_training_epoch,
                 convertor_epoch_number,
                 pretrain_path,
                 with_student_loss=True,
                 with_teacher_loss=False,
                 **kwargs):
        super(OSDDistill, self).__init__(with_student_loss, with_teacher_loss, **kwargs)
        self.collect_key = collect_key
        self.solve_number = solve_number
        self.convertor_training_epoch = convertor_training_epoch
        gpu = int(os.environ['LOCAL_RANK'])
        self.convertor_epoch_number = convertor_epoch_number
        self.convertor = DDP(Mulit_Augmentation(
            pretrain_path=pretrain_path,
            solve_number=self.solve_number
        ).cuda(gpu),
            device_ids=[gpu],
            find_unused_parameters=True )
    def train_convertor_step(self, data, optimizer):
        augment_data = dict()
        augment_data["img_metas"] = copy.deepcopy(data["img_metas"])
        data["img"].requires_grad = True
        self.distiller.set_convertor_training()
        augment_data["img"] ,augment_data["gt_bboxes"] ,augment_data["gt_labels"]\
            = self.convertor(data["img"] ,data["gt_bboxes"] ,data["gt_labels"])
        losses = dict()
        if self.with_student_loss:
            student_losses = self.distiller.exec_student_forward(
                self.architecture, augment_data)
            student_losses = add_prefix(student_losses, 'student')
            losses.update(student_losses)
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            _ = self.distiller.exec_student_forward(self.architecture, augment_data)

        if self.with_teacher_loss:
            teacher_losses = self.distiller.exec_teacher_forward(augment_data)
            teacher_losses = add_prefix(teacher_losses, 'teacher')
            losses.update(teacher_losses)
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            _ = self.distiller.exec_teacher_forward(augment_data)

        distill_losses = self.distiller.compute_distill_loss(augment_data)
        distill_losses = add_prefix(distill_losses, 'distiller')
        losses.update(distill_losses)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(augment_data['img'].data))
        self.distiller.unset_convertor_training()
        return outputs

    def train_step(self, data, optimizer):
        gt_bboxes = data["gt_bboxes"]
        gt_labels = data["gt_labels"]
        img = data["img"]
        with torch.no_grad():
            new_img, new_gt_bboxes, new_gt_labels = self.convertor.module(img,gt_bboxes,gt_labels)
        img = torch.cat([new_img,img],dim=0)
        gt_bboxes = new_gt_bboxes + gt_bboxes
        gt_labels = new_gt_labels + gt_labels
        data["gt_bboxes"] = gt_bboxes
        data["gt_labels"] = gt_labels
        data["img"] = img
        data["img_metas"] = data["img_metas"] + copy.deepcopy(data["img_metas"])
        return super(OSDDistill, self).train_step(data, optimizer)

