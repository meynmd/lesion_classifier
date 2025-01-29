# skin_classifier

## Installation
- create and activate conda environment with python=3.9
- install pytorch using following command:
`conda install pytorch::pytorch torchvision -c pytorch`
  - or use appropriate command for your platform and GPU setup:
`https://pytorch.org/get-started/locally/`
- conda install pip
- pip dependencies:
  - pandas
  - torcheval
  - tqdm
  - tensorboard
  - transformers
  - ipywidgets
  - tensorflow>=2.16


## Results
Models trained on Binary XEntropy loss for 20 epochs, with ADAM, base LR = .0002 with 2 epoch warmup and cosine decay. Initialization from ImageNet pretrained parameters. Classifier layer initizlied with Xavier.

### Baseline ResNet18 without Data Augmentations
<img width="459" alt="Screenshot 2025-01-29 at 3 25 02 PM" src="https://github.com/user-attachments/assets/5412e8cd-ad6e-4f13-a1e7-90b324b46c81" /> </br>
<img width="459" alt="Screenshot 2025-01-29 at 3 37 59 PM" src="https://github.com/user-attachments/assets/131885de-06f1-452d-bd68-a4358534ab8c" />

### Baseline ResNet18 with Albumentations

<img width="459" alt="Screenshot 2025-01-29 at 3 26 11 PM" src="https://github.com/user-attachments/assets/c5f02b98-9016-4452-97df-3d8bb151adda" />  </br>
<img width="459" alt="Screenshot 2025-01-29 at 3 26 34 PM" src="https://github.com/user-attachments/assets/3dab94a3-69c4-488a-bece-a35a6177e231" />
