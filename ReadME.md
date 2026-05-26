# Intelligent Systems and Assistive Technology project for the University of the Western Cape - COS 731 + 732 (Hons project)

## Introduction

We present Deepfake Identification alongside Federated Learning to detect synthetic images. Machine learning plays a significant role in assisting with complicated and convoluted problems that breaches human ability. The precision and consistency with which technology can reliably and easily discern the integrity of digitalized visual media is thus critical.

## Requirements

* ``` Olivetti faces dataset ```
* ``` Python 3.8 (or above) ```

## Dataset specifications

There are 400 sample images where there are 10 different images of 40 distinct subjects. The subjects vary from lighting, facial expressions of open eyes, closed eyes, smiling, not smiling, and facial details such as glasses and/or no glasses.

## Deepfake Identification system setup

It is good practice to ensure and update all packages frequently.

## Dataset setup

Please ensure you have correctly input/locate the appropriate directory where the dataset is located. By ensuring this it allows for the data to be loaded and manipluated.

## Using Google Colab

* ``` Upload to session storage ```
* ``` Create a new folder named: Face_Recognition ```
* ``` Navigate in the directory where the dataset is located ```
* ``` Navigate to "Olivetti Dataset" ```
* ``` Upload "olvetti_faces.npy" and "olivetti_faces_target.npy" ```
* ``` Move the two (2) .npy files into the Face_Recognition folder ```

## Ensure that the dataset is properly loaded by running the following code in the notebook

    import os
    mydir = r'/content/Face_Recognition'
    myfile = 'Face_Recognition'
    Face_Recognition_path = os.path.join(mydir, myfile)

To load the image data and verify the data run the following code:

    data = np.load("/content/Face_Recognition/olivetti_faces.npy")
    target = np.load("/content/Face_Recognition/olivetti_faces_target.npy")

Verify the data by running:

    print("There are {} images in the dataset".format(len(data)))
    print("There are {} unique targets in the dataset".format(len(np.unique(target))))
    print("The size of each image is: {}x{}".format(data.shape[1],data.shape[2]))

Preceeding the successful loading and verification of the dataset the following will be displayed:

* ``` Number of images in the dataset ```
* ``` Number of unique targets in the dataset ```
* ``` The size of each image in the dataset ```

## Then to simply install other libs

In terminal to:

Install Numpy for linear algebra

* ``` pip install Numpy ```

Install Pandas for data processing

* ``` pip install Pandas ```

## Machine Learning setup

Please ensure to enable Machine Learning features for testing and training purposes.

Within the code cell import the following machine learning features:

    from sklearn import metrics

Architectural Models:

    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB

Train-test split on the dataset

    from sklearn.model_selection import train_test_split

Graph Visualization, PCA for component analysis, Heatmap of Confusion Matrix

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import seaborn as sns

Cross Validation of different architectural models

    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

Saving the models as Pickle files (.plk)

    import pickle
    import joblib

## Federated Learning environment setup

Import the following libraries:

    import numpy as np
    import random
    import cv2
    import os
    from imutils import paths
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.utils import shuffle
    from sklearn.metrics import accuracy_score

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Activation
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras import backend as K

## Conclusions

B.Sc (Hons) in Computer Science for University of Western Cape, Cape Town, 7535, South Africa.

Honours project: ISAT project - Deepfake Identification alongside Federated Learning

Subject code: COS731 (Semester 1) + COS732 (Semester 2)

Student:

* Fergan van Jaarsveld - 4164150@myuwc.ac.za

Supervisors:

* Dr. Olasupo Ajayi - ooajayi@uwc.ac.za
* Hlonipani Maluleke - hhmaluleke@uwc.ac.za

Co-Supervisor:

* Prof. Antoine Bagula - abagula@uwc.ac.za
---------------------------------------------------------------------------------
# v2 - modernized detection

A modern, modular deepfake detection project combining state-of-the-art deep
learning models, traditional ML with PCA, and federated learning

## What's inside:

| Component | What it does |
|-----------|-------------|
| **Deep learning** | EfficientNet-B4, Xception, ViT-B/16 — pretrained on ImageNet, fine-tuned for deepfake detection |
| **Traditional ML + PCA** | LDA, Logistic Regression, Gaussian Naive Bayes trained on PCA-compressed face features |
| **Federated learning** | Flower-based FedAvg and FedProx simulation across N clients |
| **Gradio demo** | Browser UI — upload a photo, get a fake probability score |
| **Notebooks** | Four Jupyter notebooks walking through each component |

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a dataset

See `data/README.md` for full instructions.  The easiest option is Celeb-DF v2:

```bash
python data/scripts/download_celebdf.py   # opens the request form
# (after receiving your download link, extract the archive)

python data/scripts/prepare_dataset.py \
    --dataset celebdf \
    --root /path/to/Celeb-DF-v2 \
    --out data/celeb-df
```

### 3. Train a deep learning model

```bash
python scripts/train.py --model efficientnet --data data/celeb-df
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/efficientnet_best.pth \
    --data data/celeb-df \
    --save-plots
```

### 5. Launch the demo

```bash
python scripts/run_demo.py --checkpoint checkpoints/efficientnet_best.pth
# Open http://localhost:7860
```

---

## All scripts

| Script | What it does |
|--------|-------------|
| `scripts/train.py` | Train a deep learning model |
| `scripts/train_ml.py` | Train a traditional ML model with PCA |
| `scripts/federated_train.py` | Run federated learning simulation |
| `scripts/evaluate.py` | Evaluate any trained model on the test set |
| `scripts/run_demo.py` | Launch the Gradio web demo |

Every script has `--help` for full option descriptions.

---

## Available models

### Deep learning  (`--model <name>`)

| Name | Architecture | Notes |
|------|-------------|-------|
| `efficientnet` | EfficientNet-B4 | Best accuracy/speed tradeoff — **recommended** |
| `xception` | Xception | Original FaceForensics++ baseline, strong & fast |
| `vit` | ViT-B/16 | Vision Transformer, highest accuracy but needs more GPU |

### Traditional ML  (`--model <name>` in `train_ml.py`)

| Name | Algorithm |
|------|-----------|
| `lda` | Linear Discriminant Analysis |
| `logistic_regression` | Logistic Regression |
| `gaussian_nb` | Gaussian Naive Bayes |

**To add your own model** — see [`ADDING_MODELS.md`](ADDING_MODELS.md).

---

## Configuration

All hyperparameters live in YAML files — no code changes needed.

| File | Controls |
|------|----------|
| `config/training.yaml` | Model, dataset, epochs, LR, batch size, augmentation |
| `config/federated.yaml` | FL rounds, clients, strategy (fedavg/fedprox), local epochs |
| `config/models/*.yaml` | Per-model defaults (dropout, LR, timm name) |

CLI flags always override config file values:

```bash
python scripts/train.py --model xception --epochs 30 --batch-size 64
```

---

## Project structure

```
deepfake-detection/
├── config/                 # YAML configuration files
├── data/
│   ├── README.md           # Dataset setup instructions
│   └── scripts/            # Download and prepare scripts
├── src/deepfake/
│   ├── data/               # DeepfakeDataset, transforms
│   ├── models/             # Registry, EfficientNet, Xception, ViT
│   │   └── traditional/    # LDA, LR, GNB wrappers
│   ├── pca/                # DeepfakePCA with visualisation
│   ├── training/           # PyTorch Trainer
│   ├── federated/          # Flower client, server, utils
│   ├── evaluation/         # Metrics, confusion matrix, ROC
│   └── app/                # Gradio demo
├── notebooks/              # Jupyter walkthroughs
│   ├── 01_data_exploration.ipynb
│   ├── 02_traditional_ml_pca.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_federated_learning.ipynb
├── scripts/                # CLI entry points
├── checkpoints/            # Saved model weights (auto-created)
└── requirements.txt
```

---

## Federated learning

The FL simulation runs entirely in-process — no separate server process needed.

```bash
# FedAvg (default)
python scripts/federated_train.py --rounds 50 --clients 10

# FedProx (better for non-IID data)
python scripts/federated_train.py --strategy fedprox --rounds 50 --clients 10

# Custom model
python scripts/federated_train.py --model xception --rounds 20
```

**FedAvg vs FedProx**:
- FedAvg: Standard weighted averaging of client weights after each round.
- FedProx: Adds a proximal term (μ=0.01 by default) that keeps each client's
  weights close to the global model.  More stable when clients have very
  different data distributions.

Adjust `fedprox_mu` in `config/federated.yaml` to control how strongly
clients are pulled toward the global model.

---

## PCA in the traditional ML pipeline

PCA reduces the dimensionality of raw face images before feeding them to
classical classifiers.  The pipeline:

1. Load face images, flatten to pixel vectors (64×64×3 = 12,288 dims)
2. Fit PCA — automatically selects components to explain 95% of variance
3. Transform train and test features
4. Train LDA / LR / GNB on the compressed features

Visualisations produced by `train_ml.py`:
- `checkpoints/pca_variance.png` — cumulative explained variance curve
- `checkpoints/eigenfaces.png` — top principal components as face images

To change the number of components:
```bash
python scripts/train_ml.py --model lda --pca-components 50
```

---

## Supported datasets

| Dataset | Size | Access |
|---------|------|--------|
| **Celeb-DF v2** | ~2 GB | Short form request |
| **FaceForensics++** | ~20 GB | Form request to authors |
| **DFDC Preview** | ~10 GB | Kaggle account |

See `data/README.md` for setup instructions for each.

---

## Requirements

- Python 3.9+
- PyTorch ≥ 2.1 (`pip install torch torchvision`)
- GPU recommended for deep learning training; CPU works for traditional ML

Full list in `requirements.txt`.

Components delivered:

| Component | FIles | What it does |
|-----------|-------|--------------|
| Config system | config/training.yaml, config/federated.yaml, config/models/*.yaml | All hyperparameters in one place - no code changes needed |
| Dataset pipeline | data/scripts/prepare_dataset.py + download helpers | Supports Celeb-DF v2, FaceForensics++, DFDC; extracts face drops, writes CSV manifests |
| Model registry | model/registry.py + models/__init__.py | @register_model("name") pattern - adding a model is 3 steps, documented in ADDING_MODELS.md |
| Deep learning models | efficientnet.py, xception.py, vit.py | EfficientNet-B4, Xception, ViT-B/16 via timm, all with the same interface | 
| Traditional ML + PCA | traditional/ + pca/pipeline.py | LDA, LR, GNB wrappers; DeepfakePCA with eigenface/variance visulalisation |
| PyTorch Trainer | training/trainer.py | Train/val loops, early stopping, cosine LR, TensorBoard, checkpoint saving |
| Federated learning | federated/client.py, server.py, utils.py | Flower NumPyClient wrapping any registered model; FedAvg + FedProx |
| Evaluation | evaluation/metrics.py | Accuracy, AUC-ROC, F1, confusion matrix, ROC curve |
| Gradio demo | app/demo.py + scripts/run_demo.py | Browser UI: upload photo → face crop → fake probability |
| 4 Jupyter notebooks | notebooks/01-04 | Self-contained walkthroughs for data, ML + PCA, deep learning, FL |
| Documentation | README.md, ADDINF_MODELS.md, data/README.md | Quickstart, full model guide, dataset setup |
