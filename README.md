# skin_classifier

## Installation
- create and activate conda environment with python=3.9
- install pytorch using following command:
`conda install pytorch::pytorch torchvision -c pytorch`
  - or use appropriate command for your platform and GPU setup:
`https://pytorch.org/get-started/locally/`
- conda install pip
- pip install -r requirements.txt
  - ...or install pip dependencies manually...
    - numpy
    - pandas
    - torcheval
    - tqdm
    - tensorboard
    - transformers
    - matplotlib
    - scikit-learn
    - albumentations
    - pillow
    - ipywidgets
    - tensorflow>=2.16

## Usage
Prepare training, validation and test splits using `scripts/prepare_split_data.py`. For the current dataset, it is recommended to split the preexisting `train` data into training and validation splits. For example, to produce datalist CSV files with a 95/5 train/val split, you can run the following command:
```
 python scripts/prepare_split_data.py -d <image-root>/train -o <image-root> --do-split
```
The test set does not have to be split, of course, but for consistency of data loading, we also produce a CSV list of files to test on:
```
python scripts/prepare_split_data.py -d <image-root>/test -o <image-root>
```

Scripts for training are also found in the `scripts` directory. Run a training script with the `-h` flag to see available and required arguments. For example, to get the usage for training a ResNeXt50/32x4d using data augmentations from Albumentations, run the command, `python scripts/train_resnext50_albumentations.py -h`.

Models, data utilities, and engine code can be found under `src`.

## Results
Models trained on Binary XEntropy loss for 20 epochs, with ADAM, base LR = .0002 with 2 epoch warmup and cosine decay. Initialization from ImageNet pretrained parameters. Classifier layer initizlied with Xavier.

<img width="459" alt="Screenshot 2025-01-29 at 4 19 06 PM" src="https://github.com/user-attachments/assets/ed362860-6dea-4fc7-99c0-6d59c4a64627" />

### Baseline ResNet18 without Data Augmentations
<img width="459" alt="Screenshot 2025-01-29 at 3 40 55 PM" src="https://github.com/user-attachments/assets/da1b63f5-396b-4ab5-8933-6cecbb4ec473" />

### ResNet18 with Albumentations
<img width="459" alt="Screenshot 2025-01-29 at 3 26 34 PM" src="https://github.com/user-attachments/assets/3dab94a3-69c4-488a-bece-a35a6177e231" />

### ResNext50 with Albumentations
<img width="459" alt="Screenshot 2025-01-29 at 4 17 27 PM" src="https://github.com/user-attachments/assets/31d46cbf-1d2a-41e6-a3a2-d87ec0dce945" />
