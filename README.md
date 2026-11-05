# ASLReader-ConvNeXt

## About the Project

A PyTorch rewrite of [ASLReader](https://github.com/Lepieu/ASLReader). Instead of a small hand-built Keras CNN, it fine-tunes a [ConvNeXt-Tiny](https://arxiv.org/abs/2201.03545) network that starts from ImageNet-pretrained weights. Out of the box it reads American Sign Language (29 categories: each letter plus `del`, `nothing` and `space`), but it works for any set of image categories.

Changes from the Keras version:

- **ConvNeXt-Tiny** (torchvision) instead of two Conv2D/MaxPool layers. Pretrained weights mean much higher accuracy from far fewer epochs.
- **No hand-edited `lettertonumber` dict or `outputs` variable.** The categories are read from the folder names in `Database/`.
- **Images load lazily** from disk in batches, so large datasets like the 87k-image ASL alphabet set don't need to fit in RAM.
- **Light data augmentation** (small rotations, shifts and color changes) on the training images only.
- **The trained network is saved to a file** after every epoch, can be loaded to keep training, and can categorize new images, folders of images, or videos (frames sampled at `FPS`).
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

### Saving and loading the network

The network is saved to `--out` (default `aslreader_convnext.pt`) after every epoch, so stopping training early keeps the progress so far. The file holds the weights, the category names, the image size and how many epochs it has been trained for, so it can be loaded without the training data.

Keep training a saved network:

```bash
python photorecognition_convnext.py train --data Database --epochs 3 --resume aslreader_convnext.pt
```

The data must have the same categories the network was trained on. To use a saved network from your own code:

```python
from photorecognition_convnext import LoadNetwork, SaveNetwork, GetDevice

model, categories, epochs = LoadNetwork("aslreader_convnext.pt", GetDevice())
# ... use or train the model ...
SaveNetwork(model, categories, epochs, "my_network.pt")
```

Categorize an image, a folder, or a video:

```bash
python photorecognition_convnext.py predict path/to/image.jpg
python photorecognition_convnext.py predict path/to/folder
python photorecognition_convnext.py predict path/to/video.mp4
```

Image size, `FPS`, epochs, batch size and learning rate are constants at the top of the script, and most can also be overridden on the command line (`--help`).

## ConvNeXt V2

`photorecognition_convnextv2.py` works the same way as the ConvNeXt script, with the same `train` and `predict` commands and options, but uses [ConvNeXt V2](https://arxiv.org/abs/2301.00808) from [timm](https://github.com/huggingface/pytorch-image-models). V2 adds Global Response Normalization (GRN) layers and was pretrained with a masked-autoencoder method (FCMAE), which generally makes it more accurate than V1 at the same size. The default is ConvNeXt V2-Tiny (28M parameters, the same size as V1-Tiny), pretrained with FCMAE and fine-tuned on ImageNet-22k and ImageNet-1k.

```bash
python photorecognition_convnextv2.py train --data Database --epochs 5
python photorecognition_convnextv2.py predict path/to/image.jpg
```

Use `--arch` to pick another size, for example `convnextv2_nano.fcmae_ft_in22k_in1k` (15M parameters, faster) or `convnextv2_base.fcmae_ft_in22k_in1k` (89M parameters, more accurate). The saved file (`aslreader_convnextv2.pt`) records which size it is, so `predict` and `--resume` rebuild the right one. V1 and V2 saved files aren't interchangeable; each script only loads its own.

## The from-scratch NumPy network

`Model.py` and `photorecognition_original.py` are a rewrite of the original ASLReader's hand-built CNN. It's still written from scratch, with NumPy as the only math library and no deep learning framework, but it's much faster and the training math is fixed.

```bash
python photorecognition_original.py train --data Database --epochs 10
python photorecognition_original.py predict path/to/image.jpg   # or a folder, or a video
```

The network is saved to `aslreader_original.npz` and loaded by `predict`, or from your own code with `Model.Convolutional_NN.load(path)`. It takes the same `--limit`, `--batch-size`, `--lr` and `--out` options as the ConvNeXt script, and it also reads its categories from folder names.

**Why it was slow:** the old version looped over every pixel in Python, and each pass made a few tiny NumPy calls on a 3×3 patch. Almost all the time went to Python and NumPy call overhead rather than math. One training step took about 160 ms per image, so an epoch of the 87k-image ASL set took about 3 hours.

**What changed:**

- **No per-pixel loops.** Convolution lines up every patch as a row of a matrix ("im2col"), so a whole layer becomes one matrix multiplication. Pooling uses reshapes. Backpropagation works the same way.
- **Batches** of 32 images go through each layer together, instead of one image at a time.
- **No repeated forward pass.** Backpropagation reuses the values saved by `forward()` instead of computing them again.
- **Several filters per conv layer** (`setLayer("conv", (3, 3), 1, filters=8)`). The old version had one filter per layer, so it could only detect one kind of feature.
- **Momentum** in the weight updates, which smooths out noisy batches.

| | Old | New |
|---|---|---|
| Same network (1 filter per layer) | 161 ms/image, ~3.1 h/epoch | 2.6 ms/image, ~3 min/epoch (**63× faster**) |
| Default network (8 and 16 filters) | – | 9.8 ms/image, ~11 min/epoch |

**Training bugs fixed** (these were hurting accuracy):

- Pooling backpropagation used `height + step` instead of `height * step` and an off-by-one index, so gradients went to the wrong pixels.
- Conv backpropagation didn't multiply by the filter weights or apply ReLU's derivative when passing the gradient back.
- The output used ReLU instead of softmax before the cross-entropy loss.
- `range(1, epochs)` trained for one epoch fewer than asked.
- Weights started with a fixed scale of 0.1; they now use He initialization, scaled to each layer's size.

Every gradient has been checked numerically against the true derivative, and they match to within about 1e-10.

This network is a learning project, so expect it to be much less accurate than ConvNeXt. It's small, trains from random weights, and its final layer depends on exactly where in the image a feature appears.
