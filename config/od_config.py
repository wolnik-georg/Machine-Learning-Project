from .base_config import SWIN_PRESETS, SEED_CONFIG

# ============================================================
# Basics
# ============================================================

PROJECT_ROOT = "/home/pml17/Machine-Learning-Project"

# Model type selection for comparison experiments
MODEL_TYPE = "swin"   #Options: "swin" or "resnet"
SWIN_VARIANT = "tiny" # Options: "tiny", "small", "base", "large" (only used if MODEL_TYPE == "swin")

IMG_SIZE = 224  # pretrained image size (ImageNet standard)

# Can be overridden with a local path if needed
if MODEL_TYPE == "swin":
    CHECKPOINT = f"timm:swin_{SWIN_VARIANT}_patch4_window7_224.a1_in1k"
elif MODEL_TYPE == "resnet":
    CHECKPOINT = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth"
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

# Experiment output directory
RUN_NAME = f"cascade_mask_rcnn_{MODEL_TYPE}" + (f"_{SWIN_VARIANT}" if MODEL_TYPE == "swin" else "")
work_dir = f"{PROJECT_ROOT}/runs/{RUN_NAME}_fpn_coco"

# Training schedule

# 3x schedule: 36 epochs
MAX_EPOCHS = 12 # 1x schedule
MILESTONES = [8, 11]  # for 1x schedule
# MILESTONES = [27, 33]  # for 3x schedule

# Data loading

# effective batch size = BATCH_SIZE * ACCUMULATIVE_COUNTS * number_of_GPUs
BATCH_SIZE = 8
ACCUMULATIVE_COUNTS = 2  # gradient accumulation steps

NUM_WORKERS = 4

# COCO dataset

DATA_ROOT = f"{PROJECT_ROOT}/datasets/coco"
DATASET_TYPE = "CocoDataset"
ANN_TRAIN = f"{DATA_ROOT}/annotations/instances_train2017.json"
ANN_VAL = f"{DATA_ROOT}/annotations/instances_val2017.json"
TRAIN_PREFIX = "train2017/"
VAL_PREFIX = "val2017/"

# Optimizer

LR = 1e-4
WEIGHT_DECAY = 0.05

# Warmup

WARMUP_ITERS = 500
WARMUP_START_FACTOR = 0.001


# ============================================================
# Model
# ============================================================

# Backbone selection: either custom Swin Transformer or standard ResNet
if MODEL_TYPE == "swin":
    backbone = dict(
        type="SwinTransformerModel",     # custom Swin backbone registered in MMDet
        img_size=None,                   # allow dynamic image sizes (important for detection)
        patch_size=4,
        embed_dim=SWIN_PRESETS[SWIN_VARIANT]["embed_dim"],
        depths=SWIN_PRESETS[SWIN_VARIANT]["depths"],
        num_heads=SWIN_PRESETS[SWIN_VARIANT]["num_heads"],
        window_size=7,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        drop_path_rate=0.0,
        pretrain_img_size=IMG_SIZE,
        out_indices=(0, 1, 2, 3),         # output features from all Swin stages
        use_shifted_window=True,
        use_relative_bias=True,
        use_absolute_pos_embed=False,
        use_hierarchical_merge=False,
        use_gradient_checkpointing=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=CHECKPOINT,
        ),
    )
elif MODEL_TYPE == "resnet":
    backbone = dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),         # C2–C5 feature maps
        frozen_stages=1,                 # freeze stem + first stage
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(
            type="Pretrained",
            checkpoint=CHECKPOINT,
        ),
    )
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

# Input normalization and padding logic
data_preprocessor = dict(
    type="DetDataPreprocessor",
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    pad_size_divisor=32,
)

# Feature Pyramid Network (FPN)
if MODEL_TYPE == "swin":
    BASE_CHANNELS = backbone["embedding_dim"]
elif MODEL_TYPE == "resnet":
    BASE_CHANNELS = 256
neck = dict(
    type="FPN",
    in_channels=[BASE_CHANNELS, BASE_CHANNELS * 2, BASE_CHANNELS * 4, BASE_CHANNELS * 8],
    out_channels=256,
    num_outs=5,
)

# Region Proposal Network (RPN)
# Generates candidate object boxes from FPN features
rpn_head = dict(
    type="RPNHead",
    in_channels=256,
    feat_channels=256,
    anchor_generator=dict(
        type="AnchorGenerator",
        scales=[8],
        ratios=[0.5, 1.0, 2.0],
        strides=[4, 8, 16, 32, 64],
    ),
    bbox_coder=dict(
        type="DeltaXYWHBBoxCoder",
        target_means=[0.0, 0.0, 0.0, 0.0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
    ),
    loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
)

# Cascade RoI Head
# Three-stage refinement of bounding boxes + mask prediction
roi_head = dict(
    type="CascadeRoIHead",
    num_stages=3,
    stage_loss_weights=[1, 0.5, 0.25],   # later stages contribute less to loss

    # Shared box feature extraction
    bbox_roi_extractor=dict(
        type="SingleRoIExtractor",
        roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),

    # Progressive bbox heads with stricter IoU targets
    bbox_head=[
        dict(
            type="ConvFCBBoxHead",
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
        ),
        dict(
            type="ConvFCBBoxHead",
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.05, 0.05, 0.1, 0.1],
            ),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
        ),
        dict(
            type="ConvFCBBoxHead",
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.033, 0.033, 0.067, 0.067],
            ),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=10.0),
        ),
    ],

    # Mask head for instance segmentation
    mask_roi_extractor=dict(
        type="SingleRoIExtractor",
        roi_layer=dict(type="RoIAlign", output_size=14, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),
    mask_head=dict(
        type="FCNMaskHead",
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=80,
        loss_mask=dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
    ),
)

# Training-time assigners, samplers, and proposal settings
det_train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            ignore_iof_thr=-1,
        ),
        sampler=dict(
            type="RandomSampler",
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False,
        ),
        allowed_border=0,
        pos_weight=-1,
        debug=False,
    ),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_per_img=2000,
        nms=dict(type="nms", iou_threshold=0.7),
        min_bbox_size=0,
    ),
    rcnn=[
        dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
        dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.6,
                neg_iou_thr=0.6,
                min_pos_iou=0.6,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
        dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.7,
                min_pos_iou=0.7,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            mask_size=28,
            pos_weight=-1,
            debug=False,
        ),
    ],
)

# Inference-time thresholds and NMS settings
det_test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_per_img=1000,
        nms=dict(type="nms", iou_threshold=0.7),
        min_bbox_size=0,
    ),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.5),
        max_per_img=100,
        mask_thr_binary=0.5,
    ),
)

# Full detector definition
model = dict(
    type="CascadeRCNN",
    data_preprocessor=data_preprocessor,
    backbone=backbone,
    neck=neck,
    rpn_head=rpn_head,
    roi_head=roi_head,
    train_cfg=det_train_cfg,
    test_cfg=det_test_cfg,
)


# ============================================================
# Data / Pipelines
# ============================================================

# Training pipeline:
# - load image + annotations (bbox + masks)
# - random horizontal flip
# - AutoAugment with multi-scale resizing (and an occasional crop branch)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="AutoAugment",
        policies=[
            # standard multi-scale training (short side in [480..800], long side <= 1333)
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333),
                    ],
                    keep_ratio=True,
                )
            ],
            # resize -> random crop (can include negative crops) -> resize again
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[(400, 1333), (500, 1333), (600, 1333)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    dict(type="PackDetInputs"),
]

# Test/validation pipeline:
# Deterministic resize to the common COCO setting (short side 800, long side <= 1333)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="PackDetInputs"),
]

# Dataloaders

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img=TRAIN_PREFIX),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=DATASET_TYPE,
        data_root=DATA_ROOT,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img=VAL_PREFIX),
        pipeline=test_pipeline,
    ),
)

# Reuse val settings for test by default
test_dataloader = val_dataloader

# Evaluation:
# COCO AP for bounding boxes and instance masks
val_evaluator = dict(
    type="CocoMetric",
    ann_file=ANN_VAL,
    metric=["bbox", "segm"],
)

test_evaluator = val_evaluator

# ============================================================
# Optimizer / Schedulers / Loops
# ============================================================

# Optimizer wrapper:
# - AMP for mixed-precision training
optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    ),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),        # do not decay absolute positional embeddings
            relative_position_bias_table=dict(decay_mult=0.0),  # do not decay relative position bias
            norm=dict(decay_mult=0.0),                      # do not decay LayerNorm / BatchNorm
        )
    ),
    accumulative_counts=ACCUMULATIVE_COUNTS,               # gradient accumulation for larger effective batch size
)

# Learning rate schedulers

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=WARMUP_START_FACTOR,
        by_epoch=False,
        begin=0,
        end=WARMUP_ITERS,
    ),
    dict(
        type="MultiStepLR",
        by_epoch=True,
        milestones=MILESTONES,
        gamma=0.1,
    ),
]

# Training / validation / test loops

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=MAX_EPOCHS)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


# ============================================================
# Env / Logging / Vis
# ============================================================

# Default registry scope
default_scope = "mmdet"

# Global logging level
log_level = "INFO"
log_processor = dict(by_epoch=True, type="LogProcessor", window_size=50)

# Custom hooks:
# - NumClassCheckHook ensures dataset class count matches model heads
custom_hooks = [dict(type="NumClassCheckHook")]

# Default runtime hooks

default_hooks = dict(
    sampler_seed=dict(type="DistSamplerSeedHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),  # save every epoch
    logger=dict(type="LoggerHook", interval=50),          # log every N iterations
)

# Visualization backends (local file-based visualization)
vis_backends = [dict(type="LocalVisBackend")]

# Visualizer used for debugging predictions and ground truth
visualizer = dict(
    name="visualizer",
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
)

# Environment configuration

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
)

# Global randomness control

randomness = SEED_CONFIG

# Checkpoint handling:
# - resume: resume training inside run folder (with optimizer step)
# - load_from: start from a pretrained or previous checkpoint
resume = None
load_from = None
