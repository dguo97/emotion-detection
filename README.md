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

# Train the model
!python emotion_detection.py train --epochs 15
```

5. Once training has finished, upload a photo of a face in a new cell and classify it:

```bash
from google.colab import files
files.upload()  # select e.g. face.jpg
!python emotion_detection.py predict --image face.jpg
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
   * Input images pass through two `Conv2D` layers (32 and 64 filters) with ReLU activations to extract visual features such as edges, curves and facial shapes.
   * `MaxPooling2D` layers reduce spatial dimensions while retaining key features, making the model computationally efficient and tolerant of slight positional shifts.

2. **Classification (Fully Connected Layers):**
   * Extracted feature maps are flattened into a single feature vector.
   * A 128-unit dense layer with 50% dropout feeds a softmax output layer that gives a probability for each emotion class.

3. **Training & Optimization:**
   * **Data Augmentation:** The training set is augmented with shear, zoom and horizontal flips to reduce overfitting.
   * **Loss Function:** Categorical Cross-Entropy evaluates the difference between predicted class probabilities and ground-truth targets.
   * **Optimizer:** Adam iteratively updates network weights via backpropagation to minimize error (15 epochs by default).

4. **Inference:**
   * Press SPACE in the webcam window to grab a frame, or pass an image file. It is converted to grayscale, resized to 48x48 and classified.

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
# Train, save the model, then open the webcam for a prediction
python emotion_detection.py

# Train only
python emotion_detection.py train --epochs 20

# Predict from the webcam using a previously saved model
python emotion_detection.py predict

# Predict from an image file instead of the webcam
python emotion_detection.py predict --image face.jpg
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--train-dir` | `emotion_detection_data/train` | Training images directory |
| `--test-dir` | `emotion_detection_data/test` | Validation images directory |
| `--epochs` | `15` | Number of training epochs |
| `--batch-size` | `64` | Batch size |
| `--image` | none | Classify this image file instead of using the webcam |

Training writes `emotion_model.keras` and `class_labels.json` to the working directory.

---

## 📝 Notes

* The webcam demo and image prediction classify the whole image, not a cropped face. For better accuracy, add face detection (for example OpenCV's Haar cascades) and crop to the face before resizing.
* This is a learning project, not a production-grade emotion recogniser.

---

## 📄 License

MIT, see [LICENSE](LICENSE).
