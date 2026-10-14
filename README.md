# Emotion Detection

A small convolutional neural network (CNN) that classifies facial emotions from 48x48 grayscale images, with a webcam demo for trying it out live.

## How it works

- **Data:** images are loaded from folders (one sub-folder per emotion) and resized to 48x48 grayscale. The training set is augmented with shear, zoom and horizontal flips.
- **Model:** two `Conv2D` + `MaxPooling2D` blocks (32 and 64 filters), then a 128-unit dense layer with 50% dropout and a softmax output over the emotion classes.
- **Training:** Adam optimiser, categorical cross-entropy, 15 epochs by default.
- **Inference:** press SPACE in the webcam window to grab a frame, which is converted to grayscale, resized and classified.

## Dataset layout

Place your data in this structure (the FER-2013 dataset in image-folder form works well):

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

Class names are taken from the folder names.

## Installation (local machine)

Run these in a terminal. They will not work if pasted into a Colab cell; for Colab, jump to [Running on Google Colab](#running-on-google-colab).

```bash
git clone https://github.com/dguo97/emotion-detection.git
cd emotion-detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

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

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--train-dir` | `emotion_detection_data/train` | Training images directory |
| `--test-dir` | `emotion_detection_data/test` | Validation images directory |
| `--epochs` | `15` | Number of training epochs |
| `--batch-size` | `64` | Batch size |
| `--image` | none | Classify this image file instead of using the webcam |

Training writes `emotion_model.keras` and `class_labels.json` to the working directory.

## Running on Google Colab

Colab already ships with TensorFlow and OpenCV, so there is nothing to install. Note that Colab cannot open a local webcam window, so predictions are made on uploaded image files instead.

1. Open a new notebook at [colab.research.google.com](https://colab.research.google.com) and switch on a GPU: **Runtime > Change runtime type > GPU**.

2. Clone the repo:

   ```python
   !git clone https://github.com/dguo97/emotion-detection.git
   %cd emotion-detection
   ```

3. Get the dataset into `emotion_detection_data/` (pick one):

   **Option A: Kaggle** (the image-folder version of FER-2013). Upload your `kaggle.json` API token when prompted:

   ```python
   from google.colab import files
   files.upload()  # select kaggle.json
   !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !kaggle datasets download -d msambare/fer2013 -p emotion_detection_data --unzip
   ```

   **Option B: Google Drive.** If you keep the `train/` and `test/` folders in Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp -r /content/drive/MyDrive/emotion_detection_data .
   ```

   Copying to Colab's local disk makes training much faster than reading from Drive.

4. Train the model:

   ```python
   !python emotion_detection.py train --epochs 15
   ```

5. Classify an image. Upload a photo of a face, then run:

   ```python
   from google.colab import files
   files.upload()  # select e.g. face.jpg
   !python emotion_detection.py predict --image face.jpg
   ```

6. Optional: download the trained model to reuse elsewhere:

   ```python
   files.download('emotion_model.keras')
   files.download('class_labels.json')
   ```

Colab sessions are temporary, so save the model to Drive or download it before the runtime disconnects.

## Notes

- The webcam demo and image prediction classify the whole image, not a cropped face. For better accuracy, add face detection (for example OpenCV's Haar cascades) and crop to the face before resizing.
- This is a learning project, not a production-grade emotion recogniser.

## License

MIT, see [LICENSE](LICENSE).
