_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_2x.py',
    '../../_base_/mmdet_runtime.py'
]

custom_imports = dict(imports=['SDAKD'], allow_failed_imports=False)

# model settings
student = dict(
    type='mmdet.RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='OSDFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='OSDL1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


teacher = dict(
    type='mmdet.RetinaNet',
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_1x_coco/retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth'),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))


algorithm = dict(
    type='OSDDistill',
    solve_number=4,
    convertor_training_epoch=[6,12],
    convertor_epoch_number=1,
    pretrain_path='/home/sst/product/SDAKD/SDAKD_FOR_BASELINE/checkpoints/Augmentation',
    collect_key=['img', 'gt_bboxes', 'gt_labels'],
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='OSDTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='bbox_head.retina_cls',
                teacher_module='bbox_head.retina_cls',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_dist_roi_cls_head',
                        tau=1,
                        loss_weight=1.,
                    )
                ]),
            dict(
                student_module='neck',
                teacher_module='neck',
                losses=[
                    dict(
                        type='MSELoss',
                        name='loss_mse_fpn',
                        loss_weight=1,
                    )
                ],
                align_module=dict(
                    type='conv2d',
                    num_modules=5,
                    student_channels=256,
                    teacher_channels=256,
                )
            ),
        ]),
)
runner = dict(type='OSDBasedRunner', max_epochs=24)
find_unused_parameters = True
# fp16 = dict(loss_scale=512.)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
