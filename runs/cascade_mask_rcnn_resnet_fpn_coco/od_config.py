ACCUMULATIVE_COUNTS = 2
ANN_TRAIN = '/home/pml17/Machine-Learning-Project/datasets/coco/annotations/instances_train2017.json'
ANN_VAL = '/home/pml17/Machine-Learning-Project/datasets/coco/annotations/instances_val2017.json'
BASE_CHANNELS = 256
BATCH_SIZE = 8
CHECKPOINT = '/home/pml17/Machine-Learning-Project/trained_models/timm_resnet50.pth'
DATASET_TYPE = 'CocoDataset'
DATA_ROOT = '/home/pml17/Machine-Learning-Project/datasets/coco'
IMG_SIZE = 224
LR = 0.0001
MAX_EPOCHS = 12
MILESTONES = [
    8,
    11,
]
MODEL_TYPE = 'resnet'
NUM_WORKERS = 2
PROJECT_ROOT = '/home/pml17/Machine-Learning-Project'
RUN_NAME = 'cascade_mask_rcnn_resnet'
SEED_CONFIG = dict(deterministic=False, seed=42)
SWIN_PRESETS = dict(
    base=dict(
        depths=[
            2,
            2,
            18,
            2,
        ], embed_dim=128, num_heads=[
            4,
            8,
            16,
            32,
        ]),
    large=dict(
        depths=[
            2,
            2,
            18,
            2,
        ], embed_dim=192, num_heads=[
            6,
            12,
            24,
            48,
        ]),
    small=dict(
        depths=[
            2,
            2,
            18,
            2,
        ], embed_dim=96, num_heads=[
            3,
            6,
            12,
            24,
        ]),
    tiny=dict(depths=[
        2,
        2,
        6,
        2,
    ], embed_dim=96, num_heads=[
        3,
        6,
        12,
        24,
    ]))
SWIN_VARIANT = 'tiny'
TRAIN_PREFIX = 'train2017/'
VAL_PREFIX = 'val2017/'
WARMUP_ITERS = 500
WARMUP_START_FACTOR = 0.001
WEIGHT_DECAY = 0.05
backbone = dict(
    depth=50,
    frozen_stages=1,
    init_cfg=dict(
        checkpoint=
        '/home/pml17/Machine-Learning-Project/trained_models/timm_resnet50.pth',
        type='Pretrained'),
    norm_cfg=dict(requires_grad=True, type='BN'),
    norm_eval=True,
    num_stages=4,
    out_indices=(
        0,
        1,
        2,
        3,
    ),
    style='pytorch',
    type='ResNet')
custom_hooks = [
    dict(type='NumClassCheckHook'),
]
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_size_divisor=32,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'))
default_scope = 'mmdet'
det_test_cfg = dict(
    rcnn=dict(
        mask_thr_binary=0.5,
        max_per_img=100,
        nms=dict(iou_threshold=0.5, type='nms'),
        score_thr=0.05),
    rpn=dict(
        max_per_img=1000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_across_levels=False,
        nms_post=1000,
        nms_pre=1000))
det_train_cfg = dict(
    rcnn=[
        dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.6,
                neg_iou_thr=0.6,
                pos_iou_thr=0.6,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.7,
                neg_iou_thr=0.7,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
    ],
    rpn=dict(
        allowed_border=0,
        assigner=dict(
            ignore_iof_thr=-1,
            match_low_quality=True,
            min_pos_iou=0.3,
            neg_iou_thr=0.3,
            pos_iou_thr=0.7,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(
            add_gt_as_proposals=False,
            neg_pos_ub=-1,
            num=256,
            pos_fraction=0.5,
            type='RandomSampler')),
    rpn_proposal=dict(
        max_per_img=2000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_across_levels=False,
        nms_post=2000,
        nms_pre=2000))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(
            checkpoint=
            '/home/pml17/Machine-Learning-Project/trained_models/timm_resnet50.pth',
            type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                conv_out_channels=256,
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='ConvFCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.05,
                        0.05,
                        0.1,
                        0.1,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                conv_out_channels=256,
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='ConvFCBBoxHead'),
            dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.033,
                        0.033,
                        0.067,
                        0.067,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                conv_out_channels=256,
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=80,
                num_shared_convs=4,
                num_shared_fcs=1,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='ConvFCBBoxHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            num_classes=80,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=3,
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='CascadeRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_across_levels=False,
            nms_post=1000,
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_across_levels=False,
            nms_post=2000,
            nms_pre=2000)),
    type='CascadeRCNN')
neck = dict(
    in_channels=[
        256,
        512,
        1024,
        2048,
    ],
    num_outs=5,
    out_channels=256,
    type='FPN')
optim_wrapper = dict(
    accumulative_counts=2,
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(by_epoch=True, gamma=0.1, milestones=[
        8,
        11,
    ], type='MultiStepLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = None
roi_head = dict(
    bbox_head=[
        dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=80,
            num_shared_convs=4,
            num_shared_fcs=1,
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            roi_feat_size=7,
            type='ConvFCBBoxHead'),
        dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.05,
                    0.05,
                    0.1,
                    0.1,
                ],
                type='DeltaXYWHBBoxCoder'),
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=80,
            num_shared_convs=4,
            num_shared_fcs=1,
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            roi_feat_size=7,
            type='ConvFCBBoxHead'),
        dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.033,
                    0.033,
                    0.067,
                    0.067,
                ],
                type='DeltaXYWHBBoxCoder'),
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=10.0, type='GIoULoss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            num_classes=80,
            num_shared_convs=4,
            num_shared_fcs=1,
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            roi_feat_size=7,
            type='ConvFCBBoxHead'),
    ],
    bbox_roi_extractor=dict(
        featmap_strides=[
            4,
            8,
            16,
            32,
        ],
        out_channels=256,
        roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
        type='SingleRoIExtractor'),
    mask_head=dict(
        conv_out_channels=256,
        in_channels=256,
        loss_mask=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
        num_classes=80,
        num_convs=4,
        type='FCNMaskHead'),
    mask_roi_extractor=dict(
        featmap_strides=[
            4,
            8,
            16,
            32,
        ],
        out_channels=256,
        roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
        type='SingleRoIExtractor'),
    num_stages=3,
    stage_loss_weights=[
        1,
        0.5,
        0.25,
    ],
    type='CascadeRoIHead')
rpn_head = dict(
    anchor_generator=dict(
        ratios=[
            0.5,
            1.0,
            2.0,
        ],
        scales=[
            8,
        ],
        strides=[
            4,
            8,
            16,
            32,
            64,
        ],
        type='AnchorGenerator'),
    bbox_coder=dict(
        target_means=[
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        target_stds=[
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        type='DeltaXYWHBBoxCoder'),
    feat_channels=256,
    in_channels=256,
    loss_bbox=dict(
        beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
    loss_cls=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
    type='RPNHead')
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        data_root='/home/pml17/Machine-Learning-Project/datasets/coco',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/home/pml17/Machine-Learning-Project/datasets/coco/annotations/instances_val2017.json',
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop')
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        data_root='/home/pml17/Machine-Learning-Project/datasets/coco',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                policies=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    1333,
                                ),
                                (
                                    500,
                                    1333,
                                ),
                                (
                                    600,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='AutoAugment'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        policies=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            1333,
                        ),
                        (
                            500,
                            1333,
                        ),
                        (
                            600,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='AutoAugment'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        data_root='/home/pml17/Machine-Learning-Project/datasets/coco',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/home/pml17/Machine-Learning-Project/datasets/coco/annotations/instances_val2017.json',
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/pml17/Machine-Learning-Project/runs/cascade_mask_rcnn_resnet_fpn_coco'
