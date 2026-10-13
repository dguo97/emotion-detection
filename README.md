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

## Installation

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
```

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--train-dir` | `emotion_detection_data/train` | Training images directory |
| `--test-dir` | `emotion_detection_data/test` | Validation images directory |
| `--epochs` | `15` | Number of training epochs |
| `--batch-size` | `64` | Batch size |

Training writes `emotion_model.keras` and `class_labels.json` to the working directory.

## Notes

- The webcam demo classifies the whole frame, not a cropped face. For better accuracy, add face detection (for example OpenCV's Haar cascades) and crop to the face before resizing.
- This is a learning project, not a production-grade emotion recogniser.

## License

MIT, see [LICENSE](LICENSE).
