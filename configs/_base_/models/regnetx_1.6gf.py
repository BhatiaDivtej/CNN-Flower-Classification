# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='RegNet', arch='regnetx_1.6gf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=17,
        in_channels=912,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))
