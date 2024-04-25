# CNN Flower Classification Project 1 - Can a machine recognize images?

### Course: COMP3340
### Group Number: 22

### Group Members:

-   AGARWAL, ARYAN  3035812373
-   RAJIV, ARNAV  3035709057
-   BHATIA, DIVTEJ SINGH  3035832438
-   AGRAWAL, RAHUL  3035756555
-   GOLI, SMARAN  3035830703


This document provides the necessary steps to setup and run the CNN Flower Classification deep learning project using PyTorch, MMClassification, and a custom dataset of flower images. The project uses an RTX 1080Ti for training, but similar setups can be adjusted accordingly.

## Step 1. Installation

### Clone the Repository

First, clone the repository using SSH. Open your terminal and run:

```shell

git clone git@github.com:BhatiaDivtej/CNN-Flower-Classification.git

```

### Set Up the Conda Environment

Create a new Conda environment and activate it:

```shell

conda create -n mmcls python=3.7 -y

conda activate mmcls

```

### Install Dependencies

Install the required packages in your Conda environment:

```shell

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

pip install mmcv==1.5.0

pip install mmcv-full==1.5.0

cd CNN-Flower-Classification

pip install -e .

pip install yapf==0.40.1

```

### Verify Installation

Verify that the installation is successful by running:

```shell

python -c "import mmcls"

```

If no errors occur, the repository has been installed successfully.

## Step 2. Get Data and Prepare It Before Setting Config

### Download Data

The dataset used is the "17 category flower dataset" from the Visual Geometry Group, University of Oxford. Download and prepare the data with the following commands:

```shell

wget https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz

tar zxvf 17flowers.tgz

mkdir data

mv 17flowers data/flowers

```

### Split Data

Split the data into training and validation sets. This process assumes you have a script named `split.py` to handle the splitting:

```shell

cd data

python split.py

```

### Generate Meta Files

Generate meta files which map image paths to their respective class labels:

```shell

mkdir meta

python generate_meta.py

```

### Implement Custom Dataset Class

Create a dataset class for handling the flower dataset by adapting the `imagenet.py`:

```python

# mmclassification/mmcls/datasets/flowers.py

from .builder import DATASETS

from .base import BaseDataset

@DATASETS.register_module()

class Flowers(BaseDataset):

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    CLASSES = [

        'daffodil', 'snowdrop', 'lilyValley', 'bluebell', 'crocys', 'iris', 'tigerlily', 'tulip', 'fritillary', 'sunflower', 'daisy', 'colts foot', 'dandelion', 'cowslip', 'buttercup', 'wind flower', 'pansy'

    ]

    def load_annotations(self):

        # implementation details as provided above

        pass

# Add import in mmclassification/mmcls/datasets/__init__.py

from .flowers import Flowers

```

### Step 3: Model Configuration

#### Baseline Models

In our project, we have trained several baseline models on the flower dataset. These models include:

- VGG Series: VGG11, VGG13, VGG16, VGG19

- ShuffleNet Series: ShuffleNet v1, ShuffleNet v2

- ResNet Series: ResNet 18, ResNet50

- ResNeXt Series: ResNeXt101

- RegNetX Series: RegNetX 400mf, RegNetX 8.0gf, RegNetX 1.6gf

- AlexNet (Jupyter Notebook)

- DenseNet (Jupyter Notebook)

#### Configuration Files

The data configuration and model configurations for these models are predefined and can be found in the following directories within the project:

- **Data Config**: `configs/_base_/datasets/flowers_bs32.py`

- **Model Configs**: `configs/_base_/models/`

Each model has a specific configuration file named after the model. For example, the configuration file for VGG11 is named `vgg11_flower.py`. These files include the model-specific settings and can be found in the respective model directories under the `configs` directory.

For instance:

- VGG11: `configs/vgg/vgg11_flowers.py`

- ResNet50: `configs/resnet/resnet50_flowers.py`

- RegNetX 400mf: `configs/regnetx/regnetx400mf_flowers.py`

### Step 4: Initiating Model Training

#### Environment Setup

Before starting the training process, ensure you are in the `mmcls` Conda environment. If not already activated, you can activate it using the following command:

```shell

conda activate mmcls

```

#### Launching Training Sessions

To train the models, navigate to the `CNN-Flower-Classification` directory and execute the training command. Here's the general format to start training for any of the models:

```shell

python tools/train.py

  --config configs/[model_dir]/[model_name]_flowers.py

  --work-dir 'output/[model_name]'

```

This command specifies:

- `--config`: Path to the model's configuration file.

- `--work-dir`: Directory where the trained model checkpoints and log files will be saved.

#### Example: Training VGG11

For training the VGG11 model, you need to specify the configuration file located at `configs/vgg/vgg11_flowers.py` and set the directory for saving the model and logs to `output/vgg11`. Use the following command:

```shell

python tools/train.py

  --config configs/vgg/vgg11_flowers.py

  --work-dir 'output/vgg11'

```

#### Running Other Models

Similarly, to train other models, replace `[model_dir]` and `[model_name]` in the command with the appropriate directory and model configuration file name. For example, to train ResNet50:

```shell

python tools/train.py

  --config configs/resnet/resnet50_flowers.py

  --work-dir 'output/resnet50'
'''

#### More Models on Jupyter Notebook

##### Alexnet and DenseNet training

The folder in the main directory labeled "Alexnet and Densenet" contains a jupyter notebook containing the code
to run alexnet and densenet in the same file. Download the 17flowers folder containing the labeled fdata from

'''
https://connecthkuhk-my.sharepoint.com/personal/u3570905_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fu3570905%5Fconnect%5Fhku%5Fhk%2FDocuments%2Funi%2Fy4%2Fy4sem2%2FCOMP3340%2D%20Applied%20Deep%20Learning%2Fproject%2Fflowers%2F17flowers&ga=1
'''

and run the Jupyter notebook.



Alexnet Accuracy: 64%
Resnet Accuracy: 70.59% (self written code from scratch based on the same dataset)
Densenet Accuracy: 88.23%


### Baseline Model Training Results

Below is a summary of the Top 1 Accuracies obtained after training the baseline models on the flower classification task:

### Baseline Model Training Results

Below is a summary of the Top 1 Accuracies obtained after training the baseline models on the flower classification task:

| Model           | Top 1 Accuracy (%) |

|-----------------|--------------------|

| VGG11 |59.4|

| VGG13           |70.58|

| VGG16           |68.23|

| VGG19           |51.17|

| ShuffleNet     |5.8|

| ResNeXt 101      |61.1|

| RegNetX 400mf 	|78.23529|

| RegNetX 8.0gf   |77.05882|

| RegNetX 1.6gf   |74.11765|

This table illustrates the performance variability among different architectures, highlighting the effectiveness of specific models like the RegNetX series in this particular dataset and task.

### Step 5: Hyperparameter Tuning and Input Augmentation

#### Objective

To enhance the performance of the top-performing models, we will conduct hyperparameter tuning and input augmentation. This step is crucial for identifying the best-performing model configurations under varying training conditions.

#### Selected Models for Tuning

Based on the Top-1 and Top-5 accuracy metrics, we have selected the following models for further tuning:

- **VGG13**

- **RegNetX 400mf**

- **ResNet (18, 50)**

#### Hyperparameter Settings

We will be tuning the following hyperparameters:

- **Learning Rate**: Choices are [0.1, 0.3]

- **Momentum**: Choices are [0.7, 0.9]

- **Batch Size**: Fixed at 32

- **Optimizer**: Stochastic Gradient Descent (SGD)

- **Number of Epochs**: Fixed at 50
#### Configuration Files

We have created 9 separate training configuration files, named from "flowers_bs32_1" to "flowers_bs32_9", located under `configs/_base_/schedules/`. Each file contains different combinations of learning rate and momentum.

For each selected model, there are also 9 corresponding model configuration files reflecting the combinations of hyperparameters. These files are located in their respective directories:

- **VGG13**: `configs/vgg/vgg13_flowers_1.py`, `configs/vgg/vgg13_flowers_2.py`, ..., `configs/vgg/vgg13_flowers_9.py`

- **RegNetX 400mf and ResNet Series**: Similar naming convention applies.

#### Automated Hyperparameter Tuning Script

We have prepared a script named `hyperparameter.py` within the `CNN-Flower-Classification` directory that automates the process of training the models using different hyperparameter configurations. Here's how the script works:

```python

import subprocess

import os

START, END = 1, 10

# Uncomment the line for the model you want to train:

# model_name = "vgg13"

# model_folder = "vgg"

# model_name = "regnetx_400mf"

# model_folder = "regnet"

# model_name = "resnet18"

# model_folder = "resnet"

# model_name = "resnet50"

# model_folder = "resnet"

for index in range(START, END):

    command = f"python tools/train.py --config 'configs/{model_folder}/{model_name}_{index}.py' --work-dir 'output/{model_name}_{index}'"

    os.system(command)

```

#### Instructions for Use

1\. **Activate the `mmcls` Conda Environment**:

   Ensure that the script is run within the `mmcls` environment to have access to all necessary dependencies.

   ```shell

   conda activate mmcls

   ```

2\. **Run the Script**:

   Navigate to the script's directory and run it after uncommenting the lines corresponding to the model you wish to train.

   ```shell

   python hyperparameter.py

   ```

#### Expected Results

This script will train each selected model 9 times using different combinations of the specified hyperparameters. The results (model checkpoints and logs) will be saved in the `output` directory under respective model and configuration subdirectories. This structured approach allows for systematic evaluation of model performance across a range of settings, facilitating the selection of the optimal model configuration.


## Graph 1
\
![Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-32-2.jpg)\
\
\
## Graph 2
\
![Task 2 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-32.jpg)\
\
\
## Graph 3
\
![Task 3 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-33-2.jpg)\
\
\
## Graph 4
\
![Task 1 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-33.jpg)\
\
\
## Graph 5
\
![Task 2 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-40-2.jpg)\
\
\
## Graph 6
\
![Task 3 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-40-3.jpg)\
\
\
## Graph 7
\
![Task 1 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-40.jpg)\
\
\
## Graph 8
\
![Task 2 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-41.jpg)\
\
\
## Graph 9
\
![Task 3 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-45.jpg)\
\
\
## Graph 10
\
![Task 1 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-53-46.jpg)\
\
\
## Graph 11
\
![Task 2 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-58-45-2.jpg)\
\
\
## Graph 12
\
![Task 3 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-20-58-45.jpg)\
\
\
## Graph 13
\
![Task 2 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-21-01-39-2.jpg)\
\
\
## Graph 14
\
![Task 3 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-21-01-39.jpg)\
\
\
## Graph 15
\
![Task 1 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-21-04-12.jpg)\
\
\
## Graph 16
\
![Task 2 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-21-04-13.jpg)\
\
\
## Graph 17
\
![Task 3 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-21-44-00-2.jpg)\
\
\
## Graph 18
\
![Task 3 Graph](https://github.com/BhatiaDivtej/Image-Classification-of-Flowers/blob/main/Graph_Images/PHOTO-2024-04-22-21-44-00.jpg)\
\
\
\
\
}

}




