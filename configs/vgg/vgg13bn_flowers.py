_base_ = [
    '../_base_/models/vgg13bn_flower.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
