# optimizer, modified from cifar10_bs128.py
optimizer = dict(type='SGD', lr=0.2, momentum=0.8, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[10, 25])
runner = dict(type='EpochBasedRunner', max_epochs=50)
