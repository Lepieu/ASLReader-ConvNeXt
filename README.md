# ASLReader-ConvNeXt

## About the Project

A PyTorch rewrite of [ASLReader](https://github.com/Lepieu/ASLReader). Instead of a small hand-built Keras CNN, it fine-tunes a [ConvNeXt-Tiny](https://arxiv.org/abs/2201.03545) network that starts from ImageNet-pretrained weights. Out of the box it reads American Sign Language (29 categories: each letter plus `del`, `nothing` and `space`), but it works for any set of image categories.

Changes from the Keras version:

- **ConvNeXt-Tiny** (torchvision) instead of two Conv2D/MaxPool layers. Pretrained weights mean much higher accuracy from far fewer epochs.
- **No hand-edited `lettertonumber` dict or `outputs` variable.** The categories are read from the folder names in `Database/`.
- **Images load lazily** from disk in batches, so large datasets like the 87k-image ASL alphabet set don't need to fit in RAM.
- **Light data augmentation** (small rotations, shifts and color changes) on the training images only.
- **The trained model is saved** and can categorize new images, folders of images, or videos (frames sampled at `FPS`).
- Runs on an NVIDIA GPU (CUDA), an Apple Silicon GPU (MPS) or the CPU, whichever is available.

## Prerequisites

```bash
pip install -r requirements.txt
```

Put your images in `Database/`, one subfolder per category. The folder names become the category names. You don't need to split them into training and testing sets; the program holds back 20% for testing.

```
Database/
  A/  img1.jpg  img2.jpg ...
  B/  ...
  space/ ...
```

For ASL, the [ASL Alphabet dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) has this layout. Point `--data` at its `asl_alphabet_train` folder.

## Usage

Train (saves `aslreader_convnext.pt`):

```bash
python photorecognition_convnext.py train --data Database --epochs 5
```

For a quick test run, use `--limit 50` to take only 50 images per category. `--no-pretrained` trains from scratch.

Categorize an image, a folder, or a video:

```bash
python photorecognition_convnext.py predict path/to/image.jpg
python photorecognition_convnext.py predict path/to/folder
python photorecognition_convnext.py predict path/to/video.mp4
```

Image size, `FPS`, epochs, batch size and learning rate are constants at the top of the script, and most can also be overridden on the command line (`--help`).
