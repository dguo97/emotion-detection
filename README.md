# Emotion Detection

A TensorFlow/Keras implementation of a Convolutional Neural Network (CNN) designed to recognise facial emotions from 48x48 grayscale images, with a webcam demo for trying it out live.

---

## 🚀 How to Run in Google Colab

You can run and test this project directly in Google Colab using free GPU acceleration:

1. Open [Google Colab](https://colab.research.google.com).
2. Click **+ New notebook**.
3. Switch on a GPU: **Runtime > Change runtime type > GPU**.
4. Copy and run the following code block in a cell:

```bash
# Clone repository and enter project directory
%cd /content
!git clone https://github.com/dguo97/emotion-detection.git
%cd emotion-detection

# Download the dataset from Kaggle
import subprocess
cmd = ['kaggle', 'datasets', 'download', '-d', 'msambare/fer2013', '-p', 'emotion_detection_data', '--unzip']
if subprocess.run(cmd).returncode != 0:
    # Download failed (usually missing credentials): upload your kaggle.json API token, then retry
    from google.colab import files
    files.upload()
    !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    subprocess.run(cmd, check=True)

# Train the model (this also prints accuracy per emotion on the test set at the end)
!python emotion_detection.py train
```

5. Once training has finished, upload a photo of a face in a new cell and classify it:

```bash
from google.colab import files
files.upload()  # select e.g. face.jpg
!python emotion_detection.py predict --image face.jpg --save-crop crop.png
```

6. Optional: look at the exact 48x48 face the model saw, and re-run the per-emotion test report at any time:

```bash
from IPython.display import Image
display(Image('crop.png'))
!python emotion_detection.py evaluate
```

> **Note:** The dataset download only asks you to upload a `kaggle.json` token if the first download attempt fails, which is usually down to missing Kaggle credentials (you can create a token under Kaggle > Settings > API). TensorFlow and OpenCV come pre-installed in Colab, so there is no need to run `pip install`. Colab also cannot open a local webcam window, which is why predictions use an uploaded image file. Colab sessions are temporary, so download `emotion_model.keras` and `class_labels.json` before the runtime disconnects.

---

## 📊 Dataset Overview

The project uses the **FER-2013 facial expression dataset** in image-folder form, sourced from Kaggle:
* **Dataset Link:** [Kaggle - FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

### Dataset Features:
* **Classes:** one class per emotion (such as `angry`, `happy`, `sad`), taken from the folder names.
* **Format:** 48x48 single-channel grayscale face images.
* **Storage:** Placed inside the local `emotion_detection_data/` folder with this structure:

```
emotion_detection_data/
├── train/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   └── ...
└── test/
    ├── angry/
    ├── happy/
    ├── sad/
    └── ...
```

---

## 🧠 How It Works

The model follows a standard deep-learning pipeline for image classification:

1. **Feature Extraction (Convolutional Layers):**
   * Input images pass through three `Conv2D` layers (32, 64 and 128 filters) with ReLU activations to extract visual features such as edges, curves and facial shapes.
   * `BatchNormalization` after each convolution stabilises and speeds up training. It uses `momentum=0.9` so its running statistics keep up as the weights change; with Keras' default (0.99) the model can score well during training but predict badly afterwards, often collapsing onto a single emotion.
   * `MaxPooling2D` layers reduce spatial dimensions while retaining key features, making the model computationally efficient and tolerant of slight positional shifts.

2. **Classification (Fully Connected Layers):**
   * Extracted feature maps are flattened into a single feature vector.
   * A 256-unit dense layer with 50% dropout feeds a softmax output layer that gives a probability for each emotion class.

3. **Training & Optimization:**
   * **Data Augmentation:** The training set is augmented with shear, zoom and horizontal flips to reduce overfitting.
   * **Loss Function:** Categorical Cross-Entropy evaluates the difference between predicted class probabilities and ground-truth targets.
   * **Class Weights:** FER-2013 is imbalanced (many `happy` images, very few `disgust`), so the loss is weighted to stop rare emotions being ignored.
   * **Optimizer:** Adam iteratively updates network weights via backpropagation to minimize error. Training runs for up to 50 epochs, halving the learning rate when validation loss stalls and stopping early (keeping the best weights) when validation accuracy stops improving. Early stopping only starts after a 15-epoch warm-up, because validation scores in the first epochs are unreliable and stopping then can lock in a model that guesses a single emotion.

4. **Evaluation:**
   * After training, the model is scored on the test set: overall accuracy, accuracy per emotion, a confusion matrix and the share of predictions given to each emotion. Run it again any time with `python emotion_detection.py evaluate`.

5. **Inference:**
   * Press SPACE in the webcam window to grab a frame, or pass an image file. It is converted to grayscale, the largest face is found with OpenCV's YuNet face detector and cropped so the eyes and mouth sit where they do in the training images, then resized to 48x48 and classified. YuNet works well on close-up selfies that fill the frame. Its small model file (about 230 KB) is downloaded automatically on first use and cached in `~/.cache/emotion_detection/`. If it can't be downloaded, the script falls back to a Haar cascade, and then to the centre of the image.
   * The probability of every emotion class is printed, not just the top one, so you can see how confident the model is.

---

## 💻 Running Locally

Run these in a terminal. They will not work if pasted into a Colab cell; for Colab, see the section above.

```bash
git clone https://github.com/dguo97/emotion-detection.git
cd emotion-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then place the dataset in `emotion_detection_data/` (see above) and run:

```bash
# Train, evaluate, then open the webcam for a prediction
python emotion_detection.py

# Train (and evaluate) only
python emotion_detection.py train --epochs 40

# Per-emotion accuracy and confusion matrix for a saved model
python emotion_detection.py evaluate

# Predict from the webcam using a previously saved model
python emotion_detection.py predict

# Predict from an image file instead of the webcam
python emotion_detection.py predict --image face.jpg

# Also save the 48x48 face crop the model sees, to check it looks right
python emotion_detection.py predict --image face.jpg --save-crop crop.png
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--train-dir` | `emotion_detection_data/train` | Training images directory |
| `--test-dir` | `emotion_detection_data/test` | Validation images directory |
| `--epochs` | `50` | Maximum number of training epochs (early stopping may end it sooner) |
| `--batch-size` | `64` | Batch size |
| `--image` | none | Classify this image file instead of using the webcam |
| `--save-crop` | none | Save the 48x48 face crop the model sees to this PNG path |

Training writes `emotion_model.keras` and `class_labels.json` to the working directory.

---

## 📝 Notes

* Training images are tight 48x48 face crops, so predictions work best on a clear, front-facing, well-lit face. Faces are cropped automatically before prediction, and `--save-crop` lets you check the crop. On nine informal test selfies (five smiles, four pouts, including dark, close-up and angled shots) the happy / sad model got 8 or 9 right depending on the training run; the hard cases are dark, low-angle pouts, where it can be close to a coin flip. Results vary a little between training runs.
* The script works with any set of emotion folders. Trained on a **happy / sad** subset of FER-2013 (12,045 training and 3,021 test images), this model reached about **90.9% test accuracy** (happy 91.0%, sad 90.8%). With all seven emotions the task is much harder: expect noticeably lower accuracy (small CNNs typically land around 55-65% on FER-2013, with `happy` and `surprise` easier than `fear`, `sad` and `disgust`). That seven-emotion figure is a rough guide rather than a measured result for this model, so check your own `evaluate` output.
* **If every photo gets the same emotion:** run `python emotion_detection.py evaluate`. If the per-emotion accuracy is poor (or one emotion takes most of the predictions), the model hasn't learned well and needs more or better training. If it scores well on the test set but still fails on your own photos, use `--save-crop` to check the face crop, and try a front-facing, evenly lit photo, since FER-2013 images are small, low-quality faces and studio photos can look different to the model.
* This is a learning project, not a production-grade emotion recogniser.

---

## 📄 License

MIT, see [LICENSE](LICENSE).
