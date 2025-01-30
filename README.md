# lesion_classifier

## Installation
- create and activate conda environment with python=3.9
- install pytorch using the following command:
`conda install pytorch::pytorch torchvision -c pytorch`
  - or find the appropriate command for your platform and GPU setup:
`https://pytorch.org/get-started/locally/`
- `conda install pip`
- `pip install -r requirements.txt`
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
    - keras
- the package code for models, data, engine, etc. is found in `<repo-root>`/src/lesion_classifier, so add this to your PYTHONPATH

## Usage
Prepare training, validation and test splits using `scripts/prepare_split_data.py`. For the current dataset, it is recommended to split the preexisting `train` data into training and validation splits. For example, to produce datalist CSV files with a 95/5 train/val split, you can run the following command:
```
 python scripts/prepare_split_data.py -d <image-root>/train -o <image-root> --val-split 0.05
```
The test set does not have to be split, of course, but for consistency of data loading, we also produce a CSV list of files to test on:
```
python scripts/prepare_split_data.py -d <image-root>/test -o <image-root>
```

Scripts for training are also found in the `scripts` directory. Run a training script with the `-h` flag to see available and required arguments. For example, to get the usage for the script to train a model with Albumentations data augmentations, run the command, `scripts/train_with_albumentations.py`:
```
usage: train_with_albumentations.py [-h] --data-root DATA_ROOT --output-root OUTPUT_ROOT
                                    --model-name MODEL_NAME [--n-epochs N_EPOCHS]
                                    [--init-lr INIT_LR] [--run-name RUN_NAME]
                                    [--eval-freq EVAL_FREQ] [--save-freq SAVE_FREQ] [--cuda]
                                    [--batch-size BATCH_SIZE]
```

As noted above, models, data utilities, and engine code can be found under `src/lesion_classifier`.

## Results
<img width="474" alt="Screenshot 2025-01-30 at 3 02 45 PM" src="https://github.com/user-attachments/assets/0c2a9aec-d955-45bc-8311-c47c81242176" />
<br/>
<br/>

The following 3 models were trained on Binary XEntropy loss for 20 epochs, with ADAM, base LR = .0002 with 2 epoch warmup and cosine decay. Initialization from ImageNet pretrained parameters. Classifier layer initizlied with Xavier.
<br/>

### Baseline ResNet18 without Data Augmentations
<img width="459" alt="Screenshot 2025-01-29 at 3 40 55 PM" src="https://github.com/user-attachments/assets/da1b63f5-396b-4ab5-8933-6cecbb4ec473" />
<br/>

### ResNet18 with Albumentations
The same ResNet18 was trained with image data augmentations using the Albumentations package: random scale, translation, rotation, shear, cropping, RGB and contrast/brightness adjustment. <br/>
<img width="459" alt="Screenshot 2025-01-29 at 3 26 34 PM" src="https://github.com/user-attachments/assets/3dab94a3-69c4-488a-bece-a35a6177e231" />
<br/>

### ResNext50 with Albumentations
A ResNeXt50/32x4d was then trained using the same data augmentations as the ResNet18 with Albumentations. All hyperparameters, except the pre-cropped image size (232 vs. 256 pixels), and the network itself, were kept constant. <br/>
<img width="459" alt="Screenshot 2025-01-29 at 4 17 27 PM" src="https://github.com/user-attachments/assets/31d46cbf-1d2a-41e6-a3a2-d87ec0dce945" />
<br/>
<br/>

### Derm Foundation medical image embedding model
I then tested out a linear probe approach on a medical-imagery foundation model, Google Health's Derm Foundation (https://github.com/Google-Health/derm-foundation/tree/master). After embedding the train and test sets using Derm Foundation, I trained a logistic regression classifier and a SVM on the embeddings of training-set images. On the test-set image embeddings, performance overall was lower than with either of the fine-tuned deep networks, logistic regression (the better of the two classifiers) still hit 50% precision at 99.9% recall.
<img width="474" alt="Screenshot 2025-01-30 at 3 15 36 PM" src="https://github.com/user-attachments/assets/cf66d4ca-9945-425b-91e0-2b4a5a573d51" />
<br/>

## Discussion
For performance evaluation, I have focused heavily on the high-recall end of the PR curve, which is where such a tool would likely be useful, since false negatives could have catastrophic consequences.

High precision and recall can be achieved with a relatively lightweight ResNet18. Straightforward image data augmentations (random scale, translate, rotate, shear, crop, RGB shift and contrast/brightness adjustment) make a significant difference in performance, particularly at high-recall operating points on the curve. If we constrain ourselves to operating at thresholds that achieve 99.9% or higher recall, the ResNet18 trained with augmentations reaches 70.8% precision, compared to 52.1% for the same network trained without augmentations.

ResNeXt50 underperforms ResNet18, despite its higher capacity and being trained with the same data augmentations as the ResNet. It seems likely that a larger training dataset would be needed to take advantage of ResNeXt50's larger capacity.

Using Derm Foundation as a general medical-image feature extractor is an interesting alternative, as it requires no fine tuning; but a linear probe of the Derm Foundation embeddings underperforms all fine-tuned deep neural nets (by ~5 points F1 compared to the closest) in this experiment.

