import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from .DC import DetectionColorAugmentation
from .DS import DetectionFreezeSTN
from .AP import _apply_op


def relaxed_bernoulli(logits, temp=0.05):
    u = torch.rand_like(logits, device=logits.device)
    l = torch.log(u) - torch.log(1 - u)
    return ((l + logits) / temp).sigmoid()


def pre_tran(image, mean, std):
    _image = image.mul(torch.Tensor(std)[None, :, None, None].cuda()).add(
        torch.Tensor(mean)[None, :, None, None].cuda()
    )
    _image = _image * 255
    _image = torch.floor(_image + 0.5)
    torch.clip_(_image, 0, 255)
    _image = _image.type(torch.uint8)
    return _image


class LAMBDA_AUG(nn.Module):
    def __init__(self, lambda_function):
        """
        Args:
            lambda_function: a callable function
        Dataset:
            only applied for MS-COCO
        """
        super(LAMBDA_AUG, self).__init__()
        self.mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
        self.std = [58.395 / 255, 57.12 / 255, 57.375 / 255]
        self.tran = transforms.Normalize(mean=self.mean, std=self.std)
        self.pre_tran = lambda x: pre_tran(x, self.mean, self.std)
        self.after_tran = lambda x: self.tran(x / 255)
        self.aug = lambda_function

    def forward(self, x, boxes):
        x = self.pre_tran(x)
        x = self.aug(x)
        x = self.after_tran(x)
        return x, boxes


def _gen_cutout_coord(height, width, size):
    height_loc = random.randint(0, height - 1)
    width_loc = random.randint(0, width - 1)

    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(height, height_loc + size // 2), min(width, width_loc + size // 2))

    return upper_coord, lower_coord


class Cutout(torch.nn.Module):
    def __init__(self, size=16):
        super().__init__()
        self.size = size

    def forward(self, img, boxes):
        h, w = img.shape[-2:]
        upper_coord, lower_coord = _gen_cutout_coord(h, w, self.size)

        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = torch.ones_like(img)
        mask[..., upper_coord[0]: lower_coord[0], upper_coord[1]: lower_coord[1]] = 0
        return img * mask, boxes


class Mulit_Augmentation(nn.Module):
    # TODO: LEARNING SUB POLICIES
    LEARNING_COLOR_LIST = ["Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize"]
    NO_LEARNING_COLOR_LIST = ["Equalize", "Invert"]
    LEARNING_STN_LIST = ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]
    OTHER_LIST = ["CUTMIX"]

    def __init__(self, pretrain_path, solve_number):
        super(Mulit_Augmentation, self).__init__()
        self.len_policies = len(self.LEARNING_STN_LIST) + len(self.LEARNING_COLOR_LIST)
        self.probabilities = Parameter(
            torch.zeros(self.len_policies + len(self.NO_LEARNING_COLOR_LIST) + len(self.OTHER_LIST))
        )
        self.magnitudes = Parameter(torch.zeros(self.len_policies))
        self.pretrain_path = pretrain_path
        self.solve_number = solve_number
        concat_path = lambda x, path=self.pretrain_path: os.path.join(path, x) + ".pth"

        self.learning_color_model_list = nn.ModuleList([])
        self.learning_stn_model_list = nn.ModuleList([])
        self.nolearning_model_list = []

        for name in self.LEARNING_STN_LIST:
            path = concat_path(name)
            state_dict = torch.load(path)
            model = DetectionFreezeSTN()
            model.load_state_dict(state_dict)
            self.learning_stn_model_list.append(model)

        for name in self.LEARNING_COLOR_LIST:
            path = concat_path(name)
            state_dict = torch.load(path)
            model = DetectionColorAugmentation()
            model.load_state_dict(state_dict)
            self.learning_color_model_list.append(model)

        self._freeze_parameter()  # TODO: FREEZE

        equailize = LAMBDA_AUG(
            lambda_function=lambda _image: _apply_op(
                _image, "Equalize", 0.0, interpolation=InterpolationMode.NEAREST, fill=None
            ),
        )
        self.nolearning_model_list.append(equailize)
        invert = LAMBDA_AUG(
            lambda_function=lambda _image: _apply_op(
                _image, "Invert", 0.0, interpolation=InterpolationMode.NEAREST, fill=None
            ),
        )
        self.nolearning_model_list.append(invert)
        _cutout = Cutout(224)
        self.nolearning_model_list.append(_cutout)

    def _freeze_parameter(self):

        for parameter in self.learning_stn_model_list.parameters():
            parameter.requires_grad = False
        for parameter in self.learning_color_model_list.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _clamp(self):
        EPS = 1e-8
        self.probabilities.data = torch.clamp(self.probabilities.data, EPS, 1 - EPS)
        self.magnitudes.data = torch.clamp(self.magnitudes.data, EPS, 1 - EPS)

    def forward(self, image, boxes):
        p = torch.sigmoid(self.probabilities)
        m = torch.sigmoid(self.magnitudes)
        p = relaxed_bernoulli(p)
        len = p.shape[0]
        index = torch.randperm(len).to(image.device)
        index = index[: self.solve_number].tolist()
        color_result = []
        p_iter = 0
        m_iter = 0
        for tran in self.learning_color_model_list:
            if p_iter in index:
                _m = m[p_iter].view(-1, 1).expand(image.shape[0], -1)
                now_image, now_boxes = tran(image, _m, boxes)
                now_image = p[p_iter] * now_image + (1 - p[p_iter]) * image
                color_result.append(now_image - image)
            p_iter += 1
            m_iter += 1

        stn_result = []
        for tran in self.learning_stn_model_list:
            if p_iter in index:
                _m = m[p_iter].view(-1, 1).expand(image.shape[0], -1)
                H = tran(image, _m)
                H = p[p_iter] * H + (1 - p[p_iter]) * torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).to(H.device).expand_as(H)
                stn_result.append(H - torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).to(H.device).expand_as(H))
            p_iter += 1
            m_iter += 1

        for tran in self.nolearning_model_list:
            if p_iter in index:
                now_image, new_boxes = tran(image,boxes)
                now_image = p[p_iter] * now_image + (1 - p[p_iter]) * image
                color_result.append(now_image - image)
            p_iter += 1

        if len(stn_result)>0:
            stn_result = torch.stack(stn_result).sum(0) + \
                         torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).to(stn_result[0].device).expand_as(stn_result[0])
            image, boxes = self.forward_stn(image,stn_result,boxes)
        if len(color_result)>0:
            image = image + torch.stack(color_result).sum(0)
        return image,boxes

    def forward_stn(self, x, H, boxes):
        grid = torch.nn.functional.affine_grid(H, x.size())
        x = torch.nn.functional.grid_sample(x, grid)
        boxes = self.forward_box(boxes, H, x.shape)
        return x, boxes

    def forward_box(self, boxes, H, size):
        b, c, h, w = size
        result_boxes = []
        for i, box in enumerate(boxes):  # min_x,min_y,max_x,_max_y
            min_x, min_y, max_x, max_y = torch.split(
                box, box.shape[-1], dim=-1)
            coordinates = torch.stack([torch.stack([min_x, min_y]), torch.stack([max_x, min_y]),
                                       torch.stack([min_x, max_y]), torch.stack([max_x, max_y])])  # [4, 2, nb_bbox, 1]
            coordinates = torch.cat(
                (coordinates,
                 torch.ones(4, 1, coordinates.shape[2], 1, dtype=coordinates.dtype).to(coordinates.device)),
                dim=1)  # [4, 3, nb_bbox, 1]
            coordinates = coordinates.permute(2, 0, 1, 3)  # [nb_bbox, 4, 3, 1]
            coordinates = torch.matmul(H[i], coordinates)  # [nb_bbox, 4, 2, 1]
            coordinates = coordinates[..., 0]  # [nb_bbox, 4, 2]

            min_x, min_y = torch.min(coordinates[:, :, 0], dim=1)[0], torch.min(coordinates[:, :, 1], dim=1)[0]
            max_x, max_y = torch.max(coordinates[:, :, 0], dim=1)[0], torch.max(coordinates[:, :, 1], dim=1)[0]

            min_x, min_y = torch.clip(
                min_y, min=0, max=h), torch.clip(
                min_x, min=0, max=w)
            max_x, max_y = torch.clip(
                max_x, min=min_x, max=w), torch.clip(
                max_y, min=min_y, max=h)
            box = torch.stack([min_x, min_y, max_x, max_y],
                              dim=-1)
            mask = (box[:, 0] != box[:, 2]) & (box[:, 1] != box[:, 3])
            box = box[mask, :]
            if box.shape[0] > 0:
                result_boxes.append(box)
            else:
                raise KeyError
        return result_boxes
