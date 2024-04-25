_base_ = [
    '../_base_/models/vgg13_flower.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_bs32_7.py', '../_base_/default_runtime.py'
]
optimizer = dict(lr=0.01)
