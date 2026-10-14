"""Facial emotion detection with a small CNN.

Train a convolutional neural network on 48x48 grayscale face images, then
classify the emotion of a frame captured from your webcam.

Usage:
    python emotion_detection.py train   # train and save the model
    python emotion_detection.py predict # capture a webcam frame and classify it
    python emotion_detection.py predict --image face.jpg  # classify an image file
    python emotion_detection.py all     # train, then predict (default)
"""

import argparse
import json
import os

import cv2
import numpy as np
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (48, 48)
MODEL_PATH = "emotion_model.keras"
LABELS_PATH = "class_labels.json"


def build_generators(train_dir, test_dir, batch_size=64):
    """Create augmented training and plain test data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
    )
    return train_generator, test_generator


def build_model(num_classes):
    """Three conv/pool blocks (with batch norm) and a dense classifier head."""
    model = Sequential(
        [
            Input(shape=(*IMG_SIZE, 1)),
            Conv2D(32, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(train_dir, test_dir, epochs=15, batch_size=64):
    """Train the model and save it together with its class labels."""
    train_generator, test_generator = build_generators(
        train_dir, test_dir, batch_size
    )
    model = build_model(train_generator.num_classes)

    # FER-2013 is imbalanced (lots of "happy", very few "disgust"); weight the
    # loss so rare classes are not ignored.
    counts = np.bincount(train_generator.classes, minlength=train_generator.num_classes)
    class_weight = {i: len(train_generator.classes) / (len(counts) * c) for i, c in enumerate(counts)}

    model.fit(
        train_generator,
        class_weight=class_weight,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // test_generator.batch_size,
    )

    class_labels = list(train_generator.class_indices.keys())
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f:
        json.dump(class_labels, f)
    print(f"Saved model to {MODEL_PATH} and labels to {LABELS_PATH}")
    return model, class_labels


def load_trained():
    """Load a previously saved model and its class labels."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        raise FileNotFoundError(
            "No saved model found. Run `python emotion_detection.py train` first."
        )
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_labels = json.load(f)
    return model, class_labels


def crop_face(gray, margin=0.1):
    """Return a square crop around the largest detected face.

    The training images are tight 48x48 face crops, so the model must be given
    the same kind of input. Feeding a whole photo shrinks the face to a few
    pixels and the model falls back on its default guess (usually "sad").
    If no face is found, the largest centred square is used instead.
    """
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "This OpenCV build has no CascadeClassifier (removed in OpenCV 5). "
            "Install opencv-python<5."
        )
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    height, width = gray.shape
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(margin * max(w, h))
        side = max(w, h) + 2 * pad
        cx, cy = x + w // 2, y + h // 2
    else:
        print("Warning: no face detected, using the centre of the image.")
        side = min(height, width)
        cx, cy = width // 2, height // 2
    side = min(side, height, width)
    x0 = int(np.clip(cx - side // 2, 0, width - side))
    y0 = int(np.clip(cy - side // 2, 0, height - side))
    return gray[y0 : y0 + side, x0 : x0 + side]


def preprocess(gray):
    """Crop the face, resize to 48x48 and scale to a (1, 48, 48, 1) batch."""
    face = crop_face(gray)
    resized = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return (resized.astype("float32") / 255.0)[np.newaxis, :, :, np.newaxis]


def report(model, class_labels, gray):
    """Classify a grayscale image and print the top emotion plus all scores."""
    predictions = model.predict(preprocess(gray), verbose=0)[0]
    class_idx = int(np.argmax(predictions))
    print(f"Detected Emotion: {class_labels[class_idx]} ({predictions[class_idx] * 100:.2f}%)")
    for i in np.argsort(predictions)[::-1]:
        print(f"  {class_labels[i]:<10} {predictions[i] * 100:6.2f}%")


def predict_image(model, class_labels, image_path):
    """Classify the emotion in an image file (works without a webcam, e.g. Colab)."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    report(model, class_labels, gray)


def capture_and_predict(model, class_labels):
    """Show the webcam feed; press SPACE to classify a frame, 'q' to quit."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press SPACE to capture an image, or 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam - Press SPACE to capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            report(model, class_labels, gray)
            break
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="CNN facial emotion detection")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["train", "predict", "all"],
        help="train a model, predict from webcam, or both (default)",
    )
    parser.add_argument("--train-dir", default="emotion_detection_data/train")
    parser.add_argument("--test-dir", default="emotion_detection_data/test")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--image",
        default=None,
        help="classify this image file instead of using the webcam",
    )
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        model, class_labels = train(
            args.train_dir, args.test_dir, args.epochs, args.batch_size
        )
    else:
        model, class_labels = load_trained()

    if args.mode in ("predict", "all"):
        if args.image:
            predict_image(model, class_labels, args.image)
        else:
            capture_and_predict(model, class_labels)


if __name__ == "__main__":
    main()
