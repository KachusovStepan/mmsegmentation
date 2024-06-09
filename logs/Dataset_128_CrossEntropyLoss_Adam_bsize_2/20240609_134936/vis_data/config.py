base_dir = '/home/s0140/_scratch2/mmsegmentation'
batch_size = 2
crop_size = (
    128,
    128,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        128,
        128,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/home/s0140/_scratch2/mmsegmentation/data'
dataset_type = 'Dataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=1, max_keep_ckpts=1, type='CheckpointHook'),
    logger=dict(interval=10, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, interval=500, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
experiment_name = 'Dataset_128_CrossEntropyLoss_Adam_bsize_2'
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_interval = 10
log_level = 'INFO'
log_processor = dict(by_epoch=True)
logs_dir = '/home/s0140/_scratch2/mmsegmentation/logs'
loss = 'CrossEntropyLoss'
max_epochs = 6
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=64,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=256,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=18,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            87.68234143910625,
            101.27625783429993,
            94.20303444919348,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            128,
            128,
        ),
        std=[
            20.67239945318879,
            24.662403479389187,
            28.111042386290034,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        c1_channels=12,
        c1_in_channels=64,
        channels=128,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=512,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=5,
        type='DepthwiseSeparableASPPHead'),
    pretrained='open-mmlab://resnet18_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
num_classes = 5
num_workers = 1
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.001, type='Adam', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.001, type='Adam', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
splits = '/home/s0140/_scratch2/mmsegmentation/data/splits'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='/home/s0140/_scratch2/mmsegmentation/data/splits/val.txt',
        data_prefix=dict(img_path='images', seg_map_path='gt'),
        data_root='/home/s0140/_scratch2/mmsegmentation/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                128,
                128,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='Dataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        128,
        128,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_epochs=6, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='/home/s0140/_scratch2/mmsegmentation/data/splits/train.txt',
        data_prefix=dict(img_path='images', seg_map_path='gt'),
        data_root='/home/s0140/_scratch2/mmsegmentation/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    128,
                    128,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    128,
                    128,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='Dataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            128,
            128,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        128,
        128,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(file_client_args=dict(backend='disk'), type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='/home/s0140/_scratch2/mmsegmentation/data/splits/val.txt',
        data_prefix=dict(img_path='images', seg_map_path='gt'),
        data_root='/home/s0140/_scratch2/mmsegmentation/data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                128,
                128,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='Dataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(
        scalar_save_file='/home/s0140/_scratch2/mmsegmentation/scalars.json',
        type='LocalVisBackend'),
    dict(
        save_dir=
        '/home/s0140/_scratch2/mmsegmentation/logs/Dataset_128_CrossEntropyLoss_Adam_bsize_2',
        type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(
            scalar_save_file=
            '/home/s0140/_scratch2/mmsegmentation/scalars.json',
            type='LocalVisBackend'),
        dict(
            save_dir=
            '/home/s0140/_scratch2/mmsegmentation/logs/Dataset_128_CrossEntropyLoss_Adam_bsize_2',
            type='TensorboardVisBackend'),
    ])
work_dir = '/home/s0140/_scratch2/mmsegmentation/logs/Dataset_128_CrossEntropyLoss_Adam_bsize_2'
