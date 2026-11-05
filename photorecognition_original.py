import argparse
import os

import cv2 as cv
import numpy as np

import Model

height = 128
width = 128
FPS = 10

EPOCHS = 10
TEST_SIZE = .2
BATCH_SIZE = 32
LEARNING_RATE = 0.005

MODEL_PATH = "aslreader_original.npz"


"""
Main method
"""
def main():
    parser = argparse.ArgumentParser(description="Train or run the from-scratch NumPy CNN")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train on a folder of category subfolders")
    train.add_argument("--data", default=os.path.join(os.getcwd(), "Database"))
    train.add_argument("--epochs", type=int, default=EPOCHS)
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train.add_argument("--lr", type=float, default=LEARNING_RATE)
    train.add_argument("--limit", type=int, default=None, help="only use this many images per category (quick tests)")
    train.add_argument("--out", default=MODEL_PATH)

    predict = sub.add_parser("predict", help="categorize an image, a folder of images, or a video")
    predict.add_argument("path")
    predict.add_argument("--model", default=MODEL_PATH)

    args = parser.parse_args()
    if args.command == "train":
        Train(args)
    else:
        Predict(args)


def Train(args):
    print("Loading data...")
    (images, numbered_labels, categories) = load_data(args.data, args.limit)
    print(len(images), "images,", len(categories), "categories:", categories)

    print("Splitting data...")
    one_hot = np.eye(len(categories))[numbered_labels]
    indices = np.random.default_rng(0).permutation(len(images))
    split = int(len(images) * (1 - TEST_SIZE))
    (train_idx, test_idx) = (indices[:split], indices[split:])

    print("Prepping neural network...")
    model = PrepNeuralNetwork(len(categories))

    print("Training...")
    model.train(images[train_idx], one_hot[train_idx], images[test_idx], one_hot[test_idx],
                args.epochs, args.lr, args.batch_size)

    model.save(args.out, categories)
    print("Saved network to", args.out)


def Predict(args):
    (model, categories) = Model.Convolutional_NN.load(args.model)

    if os.path.isdir(args.path):
        names = sorted(f for f in os.listdir(args.path) if not f.startswith("."))
        frames = [cv.imread(os.path.join(args.path, f)) for f in names]
    elif os.path.splitext(args.path)[1].lower() in (".mp4", ".mov", ".avi", ".mkv"):
        frames = OpenVideo(args.path)
        names = [f"frame {i}" for i in range(len(frames))]
    else:
        names = [os.path.basename(args.path)]
        frames = [cv.imread(args.path)]

    readable = [(name, TransformImage(frame)) for (name, frame) in zip(names, frames) if frame is not None]
    probabilities = model.predict(np.array([img for (_, img) in readable]))

    for ((name, _), probs) in zip(readable, probabilities):
        print(f"{name} -> {categories[probs.argmax()]} ({probs.max():.0%})")


"""
Loads images to memory as grayscale uint8, where every subfolder of path is a category.
Categories are numbered alphabetically by folder name, so nothing needs to be edited by hand.
"""
def load_data(path, limit=None):

    data = list()
    numbered_labels = list()
    categories = sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))

    for (number, category) in enumerate(categories):
        files = sorted(f for f in os.listdir(os.path.join(path, category)) if not f.startswith("."))
        for file in files[:limit]:
            img = cv.imread(os.path.join(path, category, file))
            if img is None:
                continue

            data.append(TransformImage(img))
            numbered_labels.append(number)

    return (np.array(data), np.array(numbered_labels), categories)


"""
Returns images frame by frame from video
"""
def OpenVideo(path):
    frames = list()

    video = cv.VideoCapture(path)

    original_fps = video.get(cv.CAP_PROP_FPS) or FPS
    skip = max(1, round(original_fps / min(FPS, original_fps))) #skips frames to make the video run at a certain FPS

    while video.isOpened():
        frameNumber = int(video.get(cv.CAP_PROP_POS_FRAMES))

        ret, frame = video.read()
        if ret == False:
            break

        frames.append(frame)

        #next frame = current frame + skip
        video.set(cv.CAP_PROP_POS_FRAMES, frameNumber + skip)

    video.release()

    return frames


def TransformImage(img):
    img = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR) #Resize to height x width size
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Black and White

    return img


"""
Builds the network to process the images into [outputs] different categories
"""
def PrepNeuralNetwork(outputs):

    layers = [
        Model.setLayer("conv", (3, 3), 1, filters=8),
        Model.setLayer("pool", (2, 2), 2),
        Model.setLayer("conv", (3, 3), 1, filters=16),
        Model.setLayer("pool", (2, 2), 2)
    ]

    return Model.Convolutional_NN((height, width), outputs, layers)


if __name__ == "__main__":
    main()
