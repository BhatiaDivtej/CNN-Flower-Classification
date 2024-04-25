import subprocess
import os

START, END = 1, 10
# model_name = "vgg13"
# model_folder = "vgg"

# model_name = "regnetx_400mf_b32x8"
# model_folder = "regnet"

# model_name = "resnet18_flowers_bs128"
# model_folder = "resnet"

model_name = "resnet50_b32x8_imagenet"
model_folder = "resnet"

for index in range(START, END):
    command = f"python tools/train.py --config 'configs/{model_folder}/{model_name}_{index}.py' --work-dir 'output/{model_name}_{index}'"
    os.system(command)