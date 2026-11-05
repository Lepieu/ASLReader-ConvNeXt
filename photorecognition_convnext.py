import argparse
import os

import cv2 as cv
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

height = 128
width = 128
FPS = 10

EPOCHS = 10
TEST_SIZE = .2
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# ImageNet statistics, which the pretrained ConvNeXt weights expect
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MODEL_PATH = "aslreader_convnext.pt"


"""
Main method
"""
def main():
    parser = argparse.ArgumentParser(description="Train or run a ConvNeXt image classifier (ASL alphabet by default)")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train on a folder of category subfolders")
    train.add_argument("--data", default=os.path.join(os.getcwd(), "Database"))
    train.add_argument("--epochs", type=int, default=EPOCHS)
    train.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train.add_argument("--lr", type=float, default=LEARNING_RATE)
    train.add_argument("--no-pretrained", action="store_true", help="start from random weights instead of ImageNet")
    train.add_argument("--limit", type=int, default=None, help="only use this many images per category (quick tests)")
    train.add_argument("--resume", default=None, help="keep training a network saved earlier instead of starting fresh")
    train.add_argument("--out", default=MODEL_PATH, help="file to save the network to (written after every epoch)")

    predict = sub.add_parser("predict", help="categorize an image, a folder of images, or a video")
    predict.add_argument("path")
    predict.add_argument("--model", default=MODEL_PATH)

    args = parser.parse_args()
    if args.command == "train":
        Train(args)
    else:
        Predict(args)


"""
Trains the network and saves it, along with the category names, to args.out
"""
def Train(args):
    device = GetDevice()
    print("Using device:", device)

    print("Loading data...")
    (train_loader, test_loader, categories) = load_data(args.data, args.batch_size, args.limit)
    print(len(categories), "categories:", categories)

    if args.resume:
        print("Loading network from", args.resume)
        (model, saved_categories, epochs_done) = LoadNetwork(args.resume, device)
        if saved_categories != categories:
            raise SystemExit(f"{args.resume} was trained on categories {saved_categories}, but the data has {categories}")
    else:
        print("Prepping neural network...")
        model = PrepNeuralNetwork(len(categories), pretrained=not args.no_pretrained).to(device)
        epochs_done = 0
    print(sum(p.numel() for p in model.parameters()), "parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print("Training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0

        for (images, labels) in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)

        (test_loss, test_acc) = Evaluate(model, test_loader, loss_fn, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {total_loss / seen:.4f} - accuracy: {correct / seen:.4f}"
              f" - test_loss: {test_loss:.4f} - test_accuracy: {test_acc:.4f}")

        # save every epoch so stopping early doesn't lose the training done so far
        SaveNetwork(model, categories, epochs_done + epoch + 1, args.out)

    print("Saved network to", args.out)


"""
Saves the network's weights along with the category names, so it can be loaded without the training data
"""
def SaveNetwork(model, categories, epochs, path):
    torch.save({
        "state_dict": model.state_dict(),
        "categories": categories,
        "size": (height, width),
        "epochs": epochs,
    }, path)


"""
Loads a network saved by SaveNetwork
returns (model, categories, epochs trained so far)
"""
def LoadNetwork(path, device):
    checkpoint = torch.load(path, map_location=device)
    categories = checkpoint["categories"]

    if tuple(checkpoint["size"]) != (height, width):
        raise SystemExit(f"{path} was trained on {checkpoint['size']} images, but height x width is {(height, width)}")

    model = PrepNeuralNetwork(len(categories), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return (model.to(device), categories, checkpoint.get("epochs", 0))


def Evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0

    with torch.no_grad():
        for (images, labels) in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)

    return (total_loss / max(seen, 1), correct / max(seen, 1))


"""
Categorizes an image, every image in a folder, or the frames of a video using a saved model
"""
def Predict(args):
    device = GetDevice()
    (model, categories, _) = LoadNetwork(args.model, device)
    model.eval()

    if os.path.isdir(args.path):
        names = sorted(f for f in os.listdir(args.path) if not f.startswith("."))
        frames = [cv.imread(os.path.join(args.path, f)) for f in names]
    elif os.path.splitext(args.path)[1].lower() in (".mp4", ".mov", ".avi", ".mkv"):
        frames = OpenVideo(args.path)
        names = [f"frame {i}" for i in range(len(frames))]
    else:
        names = [os.path.basename(args.path)]
        frames = [cv.imread(args.path)]

    tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    with torch.no_grad():
        for (name, frame) in zip(names, frames):
            if frame is None:
                print(name, "-> could not read")
                continue
            img = cv.cvtColor(TransformImage(frame), cv.COLOR_BGR2RGB)
            probs = model(tensor(img).unsqueeze(0).to(device)).softmax(1)[0]
            confidence, index = probs.max(0)
            print(f"{name} -> {categories[index]} ({confidence.item():.0%})")


"""
Loads images from path, where every subfolder is a category, and splits them into training and testing sets.
Categories are numbered alphabetically by folder name, so nothing needs to be edited by hand.
"""
def load_data(path, batch_size, limit=None):

    train_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # two views of the same folder so the test images don't get augmented
    train_set = datasets.ImageFolder(path, transform=train_transform)
    test_set = datasets.ImageFolder(path, transform=test_transform)

    indices = list(range(len(train_set)))
    if limit:
        counts = {}
        indices = list()
        for (i, (_, label)) in enumerate(train_set.samples):
            counts[label] = counts.get(label, 0) + 1
            if counts[label] <= limit:
                indices.append(i)

    generator = torch.Generator().manual_seed(0)
    indices = [indices[i] for i in torch.randperm(len(indices), generator=generator)]
    split = int(len(indices) * (1 - TEST_SIZE))

    workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(Subset(train_set, indices[:split]), batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(Subset(test_set, indices[split:]), batch_size=batch_size, num_workers=workers)

    return (train_loader, test_loader, train_set.classes)


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
    return cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR) #Resize to height x width size


"""
Builds a ConvNeXt-Tiny network and swaps its last layer for one with [outputs] categories
"""
def PrepNeuralNetwork(outputs, pretrained=True):

    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.convnext_tiny(weights=weights)

    # classifier is LayerNorm -> Flatten -> Linear(768, 1000); replace the Linear
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, outputs)
    return model


def GetDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
