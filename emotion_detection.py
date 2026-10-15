"""Facial emotion detection with a small CNN.

Train a convolutional neural network on 48x48 grayscale face images, then
classify the emotion of a webcam frame or an image file.

Usage:
    python emotion_detection.py train      # train, save the model, then evaluate it
    python emotion_detection.py evaluate   # per-emotion accuracy on the test set
    python emotion_detection.py predict    # capture a webcam frame and classify it
    python emotion_detection.py predict --image face.jpg  # classify an image file
    python emotion_detection.py predict --image face.jpg --save-crop crop.png
    python emotion_detection.py all        # train, evaluate, then predict (default)
"""

import argparse
import json
import os
import urllib.request

import cv2
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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


def build_generators(train_dir, test_dir, batch_size=64, classes=None):
    """Create augmented training and plain test data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        classes=classes,
    )
    return train_generator, build_test_generator(test_dir, batch_size, classes)


def build_test_generator(test_dir, batch_size=64, classes=None):
    """Plain (unaugmented, unshuffled) generator so predictions line up with labels."""
    return ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        classes=classes,
        shuffle=False,
    )


def build_model(num_classes):
    """Three conv/pool blocks (with batch norm) and a dense classifier head.

    momentum=0.9 (instead of Keras' default 0.99) lets batch norm's running
    statistics keep up with the weights as they change. With the default, the
    statistics can lag behind and the model scores well while training but
    predicts badly afterwards (often collapsing onto one emotion).
    """
    model = Sequential(
        [
            Input(shape=(*IMG_SIZE, 1)),
            Conv2D(32, (3, 3), padding="same", activation="relu"),
            BatchNormalization(momentum=0.9),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(momentum=0.9),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(momentum=0.9),
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


def train(train_dir, test_dir, epochs=80, batch_size=64, classes=None):
    """Train the model and save it together with its class labels."""
    train_generator, test_generator = build_generators(
        train_dir, test_dir, batch_size, classes
    )
    model = build_model(train_generator.num_classes)

    # FER-2013 is imbalanced (lots of "happy", very few "disgust"); weight the
    # loss so rare classes are not ignored. The usual "balanced" weights give
    # disgust ~9x the weight of an average class, which makes early training
    # swing wildly towards it, so the square root is used to soften them.
    counts = np.bincount(train_generator.classes, minlength=train_generator.num_classes)
    class_weight = {
        i: float(np.sqrt(len(train_generator.classes) / (len(counts) * c)))
        for i, c in enumerate(counts)
    }

    callbacks = [
        # Stop when validation accuracy stops improving and keep the best weights.
        # start_from_epoch gives training a warm-up: in the first epochs the batch
        # norm statistics are still settling, so validation scores are unreliable
        # and stopping early can lock in a model that guesses a single emotion.
        EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
            start_from_epoch=15,
            verbose=1,
        ),
        # Lower the learning rate when validation loss plateaus.
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    # Do not pass steps_per_epoch / validation_steps. With samples // batch_size,
    # one batch is left over each pass, so every second epoch trained on a single
    # batch ("input ran out of data") and half the epochs were wasted. Letting
    # Keras use the generator's own length gives every epoch a full pass.
    model.fit(
        train_generator,
        class_weight=class_weight,
        epochs=epochs,
        validation_data=test_generator,
        callbacks=callbacks,
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


def evaluate(model, class_labels, test_dir, batch_size=64):
    """Print overall accuracy, per-emotion accuracy and a confusion matrix.

    This shows whether the model has really learned the emotions. If it scores
    well here but fails on your own photos, the problem is the photos (lighting,
    angle, face crop); if it scores badly here, the problem is the training.
    """
    test_generator = build_test_generator(test_dir, batch_size, list(class_labels))
    test_labels = list(test_generator.class_indices.keys())
    if test_labels != list(class_labels):
        raise ValueError(
            f"Class folders {test_labels} do not match the model's labels {class_labels}."
        )

    probabilities = model.predict(test_generator, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = test_generator.classes

    n = len(class_labels)
    matrix = np.zeros((n, n), dtype=int)
    for a, p in zip(actual, predicted):
        matrix[a, p] += 1

    print(f"\nTest accuracy: {np.mean(predicted == actual) * 100:.2f}% ({len(actual)} images)")
    print("\nAccuracy per emotion (recall):")
    for i, label in enumerate(class_labels):
        total = matrix[i].sum()
        print(f"  {label:<10} {matrix[i, i] / max(total, 1) * 100:6.2f}%  ({total} images)")

    print("\nConfusion matrix (rows = true emotion, columns = predicted):")
    print("  " + " " * 10 + "".join(f"{label[:7]:>8}" for label in class_labels))
    for i, label in enumerate(class_labels):
        print(f"  {label:<10}" + "".join(f"{v:>8}" for v in matrix[i]))

    share = np.bincount(predicted, minlength=n) / len(predicted) * 100
    print("\nShare of all predictions per emotion:")
    for i, label in enumerate(class_labels):
        print(f"  {label:<10} {share[i]:6.2f}%")


YUNET_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_PATH = os.path.join(
    os.path.expanduser("~"), ".cache", "emotion_detection", "face_detection_yunet_2023mar.onnx"
)


def get_yunet_model():
    """Return the path to the YuNet face detector model, downloading it on first use.

    The model is small (about 230 KB). Returns None if it cannot be downloaded,
    in which case the caller falls back to a Haar cascade.
    """
    if os.path.exists(YUNET_PATH) and os.path.getsize(YUNET_PATH) > 100_000:
        return YUNET_PATH
    partial = YUNET_PATH + ".part"
    try:
        os.makedirs(os.path.dirname(YUNET_PATH), exist_ok=True)
        urllib.request.urlretrieve(YUNET_URL, partial)
        if os.path.getsize(partial) < 100_000:
            raise OSError("the downloaded file is too small to be the model")
        os.replace(partial, YUNET_PATH)
        return YUNET_PATH
    except Exception as error:  # offline, blocked, read-only disk, ...
        if os.path.exists(partial):
            os.remove(partial)
        print(f"Warning: could not download the YuNet face detector ({error}).")
        return None


def square_crop(gray, cx, cy, side):
    """Crop a square of the given size centred on (cx, cy), kept inside the image."""
    height, width = gray.shape
    side = int(min(side, height, width))
    x0 = int(np.clip(cx - side / 2, 0, width - side))
    y0 = int(np.clip(cy - side / 2, 0, height - side))
    return gray[y0 : y0 + side, x0 : x0 + side]


def crop_face_yunet(image):
    """Crop the largest face found by the YuNet detector, or return None.

    YuNet copes with faces of any size, including close-up selfies that fill
    the frame, which the older Haar cascade often gets wrong. The crop is placed
    using the eye and mouth positions so the face is framed like the FER-2013
    training images (eyes about a third of the way down, mouth about three
    quarters of the way down).
    """
    if not hasattr(cv2, "FaceDetectorYN"):
        return None
    model_path = get_yunet_model()
    if model_path is None:
        return None

    height, width = image.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    small = image
    if scale < 1:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    detector = cv2.FaceDetectorYN.create(
        model_path, "", (small.shape[1], small.shape[0]), 0.5, 0.3, 5
    )
    _, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        return None

    face = max(faces, key=lambda f: f[2] * f[3]) / scale
    eye_x = (face[4] + face[6]) / 2
    eye_y = (face[5] + face[7]) / 2
    mouth_y = (face[11] + face[13]) / 2
    gap = max(mouth_y - eye_y, 1.0)  # eye-to-mouth distance sets the crop size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return square_crop(gray, eye_x, eye_y + 0.4 * gap, 2.6 * gap)


def crop_face_haar(gray, margin=0.1):
    """Fallback: crop the largest face found by a Haar cascade, or return None."""
    if not hasattr(cv2, "CascadeClassifier"):  # removed in OpenCV 5
        return None
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    side = max(w, h) + 2 * int(margin * max(w, h))
    return square_crop(gray, x + w / 2, y + h / 2, side)


def crop_face(image):
    """Return a grayscale face crop from a BGR image.

    The training images are tight 48x48 face crops, so the model must be given
    the same kind of input. Feeding a whole photo shrinks the face to a few
    pixels and distorts non-square images. Tries the YuNet detector first, then
    a Haar cascade, then the largest centred square.
    """
    crop = crop_face_yunet(image)
    if crop is not None:
        return crop

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = crop_face_haar(gray)
    if crop is not None:
        print("Note: using the older Haar face detector (less reliable on close-ups).")
        return crop

    print("Warning: no face detected, using the centre of the image.")
    height, width = gray.shape
    return square_crop(gray, width / 2, height / 2, min(height, width))


def preprocess(image, save_crop=None):
    """Crop the face, resize to 48x48 and scale to a (1, 48, 48, 1) batch.

    image is a BGR (colour) image. If save_crop is a path, the 48x48 face the
    model sees is saved there (enlarged) so you can check it looks right.
    """
    face = crop_face(image)
    resized = cv2.resize(face, IMG_SIZE, interpolation=cv2.INTER_AREA)
    if save_crop:
        enlarged = cv2.resize(resized, (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_crop, enlarged)
        print(f"Saved the 48x48 crop the model sees to {save_crop}")
    return (resized.astype("float32") / 255.0)[np.newaxis, :, :, np.newaxis]


def classify(model, class_labels, image, save_crop=None):
    """Print the predicted emotion and the probability of every class."""
    probabilities = model.predict(preprocess(image, save_crop), verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    best = order[0]
    print(f"Detected Emotion: {class_labels[best]} ({probabilities[best] * 100:.2f}%)")
    print("All probabilities:")
    for i in order:
        print(f"  {class_labels[i]:<10} {probabilities[i] * 100:6.2f}%")


def predict_image(model, class_labels, image_path, save_crop=None):
    """Classify the emotion in an image file (works without a webcam, e.g. Colab)."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    classify(model, class_labels, image, save_crop)


def capture_and_predict(model, class_labels, save_crop=None):
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
            classify(model, class_labels, frame, save_crop)
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
        choices=["train", "evaluate", "predict", "all"],
        help="train, evaluate on the test set, predict, or train+evaluate+predict (default)",
    )
    parser.add_argument("--train-dir", default="emotion_detection_data/train")
    parser.add_argument("--test-dir", default="emotion_detection_data/test")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        metavar="EMOTION",
        help="train on only these emotion folders, e.g. --classes happy sad "
        "(default: every folder found)",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--image",
        default=None,
        help="classify this image file instead of using the webcam",
    )
    parser.add_argument(
        "--save-crop",
        default=None,
        help="save the 48x48 face crop the model sees to this PNG path",
    )
    args = parser.parse_args()

    if args.mode in ("train", "all"):
        model, class_labels = train(
            args.train_dir, args.test_dir, args.epochs, args.batch_size, args.classes
        )
    else:
        model, class_labels = load_trained()

    if args.mode in ("train", "evaluate", "all"):
        evaluate(model, class_labels, args.test_dir, args.batch_size)

    if args.mode in ("predict", "all"):
        if args.image:
            predict_image(model, class_labels, args.image, args.save_crop)
        else:
            capture_and_predict(model, class_labels, args.save_crop)


if __name__ == "__main__":
    main()
