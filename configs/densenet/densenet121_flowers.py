_base_ = [
    '../_base_/models/densenet121_flower.py',
    '../_base_/datasets/flowers_bs32.py',
    '../_base_/schedules/flowers_bs32.py',
    '../_base_/default_runtime.py',
]

# dataset settings
train_dataloader = dict(batch_size=256)

# schedule settings
train_cfg = dict(by_epoch=True, max_epochs=10)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# base_batch_size = (4 GPUs) x (256 samples per GPU)
auto_scale_lr = dict(base_batch_size=1024)
