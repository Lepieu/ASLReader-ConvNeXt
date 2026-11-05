import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


"""
A convolutional neural network written from scratch with NumPy.

Every layer works on a whole batch of images at once, shaped (batch, channels, height, width),
so the per-pixel work happens inside NumPy's compiled code instead of Python for loops.
"""
class Convolutional_NN:


    """Initializes each layer with small random values to be changed during training

    :param input_shape: (height, width) of the grayscale input images
    :param outputs: number of categories
    :param layers: list of layers made with setLayer
    """
    def __init__(self, input_shape, outputs, layers):

        self.numOutputs = outputs
        self.input_shape = input_shape
        self.layers = layers

        #the fully connected layer's size depends on how much the conv and pool layers shrink the image
        channels = 1
        shape = input_shape
        for layer in layers:
            if layer['type'] == 'conv':
                (filterHeight, filterWidth) = layer['shape']
                fanIn = channels * filterHeight * filterWidth
                #He initialization: keeps the signal from shrinking or exploding through relu layers
                layer['weight'] = np.random.randn(layer['filters'], channels, filterHeight, filterWidth) * np.sqrt(2 / fanIn)
                layer['bias'] = np.zeros(layer['filters'])
                channels = layer['filters']
            elif layer['type'] == 'pool' and layer['shape'][0] != layer['step']:
                raise ValueError("pool layers need step equal to their size, e.g. setLayer('pool', (2, 2), 2)")

            shape = shrinkLayer(layer, shape)

        fcSize = channels * shape[0] * shape[1]
        self.fcLayer = setLayer(type="flattened", shape=(fcSize, 1),
                                weight=np.random.randn(outputs, fcSize) * np.sqrt(1 / fcSize),
                                bias=np.zeros(outputs))


    """Runs a batch of images through the network
    :param input: images shaped (batch, height, width) or (batch, 1, height, width)
    :returns: probability of each category, shaped (batch, outputs)
    """
    def forward(self, input):

        current_input = input.reshape(input.shape[0], 1, *self.input_shape) if input.ndim == 3 else input

        for layer in self.layers:
            layer['input'] = current_input
            if layer['type'] == 'conv':
                current_input = relu(self.convolution(current_input, layer))
                layer['output'] = current_input
            elif layer['type'] == 'pool':
                current_input = self.pooling(current_input, layer)

        #flattens for the fully connected layer (remembering the shape to unflatten during backpropagation)
        self.fcLayer['inputShape'] = current_input.shape
        self.fcLayer['input'] = current_input.reshape(current_input.shape[0], -1)
        scores = self.fcLayer['input'] @ self.fcLayer['weight'].T + self.fcLayer['bias']

        return softmax(scores)


    """Applies each filter to each group of pixels on an image

    Instead of looping over every pixel, it lines up every filter-sized patch of the image as one row of a
    matrix ("im2col"), so the whole convolution becomes a single matrix multiplication.
    """
    def convolution(self, input, layer):
        step = layer['step']
        weight = layer['weight']
        (filters, channels, filterHeight, filterWidth) = weight.shape

        #every filterHeight x filterWidth patch: (batch, channels, outHeight, outWidth, filterHeight, filterWidth)
        patches = sliding_window_view(input, (filterHeight, filterWidth), axis=(2, 3))[:, :, ::step, ::step]
        (batch, _, outputHeight, outputWidth, _, _) = patches.shape

        #one row per output pixel, one column per weight it gets multiplied by
        cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(batch * outputHeight * outputWidth, -1)
        layer['cols'] = cols #saved for backpropagation

        output = cols @ weight.reshape(filters, -1).T + layer['bias']
        return output.reshape(batch, outputHeight, outputWidth, filters).transpose(0, 3, 1, 2)


    """Keeps the largest value in each size x size block"""
    def pooling(self, input, layer):
        size = layer['shape'][0]
        (batch, channels, inputHeight, inputWidth) = input.shape
        (outputHeight, outputWidth) = (inputHeight // size, inputWidth // size)

        #crop leftover rows/columns, then group pixels into blocks: (batch, channels, outHeight, outWidth, size*size)
        blocks = input[:, :, :outputHeight * size, :outputWidth * size]
        blocks = blocks.reshape(batch, channels, outputHeight, size, outputWidth, size).transpose(0, 1, 2, 4, 3, 5)
        blocks = blocks.reshape(batch, channels, outputHeight, outputWidth, size * size)

        layer['argmax'] = blocks.argmax(axis=-1) #which pixel won, for backpropagation
        return blocks.max(axis=-1)


    """
    BACKPROPAGATION:

    Uses the values saved during forward() to work out how much each weight contributed to the loss,
    then moves every weight a small step in the direction that lowers it.

    Helper methods: backprop_conv, backprop_pool
    """
    def backpropagation(self, y, predictions, learningRate=0.005):
        batch = y.shape[0]

        #softmax followed by cross entropy has a simple combined gradient
        lossGradient = (predictions - y) / batch

        fcInput = self.fcLayer['input']
        inputChange = lossGradient @ self.fcLayer['weight']
        self.update(self.fcLayer, 'weight', lossGradient.T @ fcInput, learningRate)
        self.update(self.fcLayer, 'bias', lossGradient.sum(axis=0), learningRate)

        inputChange = inputChange.reshape(self.fcLayer['inputShape'])

        for layer in reversed(self.layers):
            if layer['type'] == 'conv':
                (weightChange, biasChange, inputChange) = self.backprop_conv(inputChange, layer)
                self.update(layer, 'weight', weightChange, learningRate)
                self.update(layer, 'bias', biasChange, learningRate)
            elif layer['type'] == 'pool':
                inputChange = self.backprop_pool(inputChange, layer)

        return inputChange


    """Gradient descent with momentum: each step keeps part of the previous step's direction,
    which smooths out the zig-zagging you get from noisy batches"""
    def update(self, layer, key, gradient, learningRate, momentum=0.9):
        velocity = momentum * layer.get('velocity_' + key, 0) - learningRate * gradient
        layer['velocity_' + key] = velocity
        layer[key] += velocity


    def backprop_conv(self, lossGradient, layer):
        step = layer['step']
        weight = layer['weight']
        (filters, channels, filterHeight, filterWidth) = weight.shape
        inputShape = layer['input'].shape
        (batch, _, outputHeight, outputWidth) = lossGradient.shape

        #relu only let positive values through, so only those get a gradient
        lossGradient = lossGradient * (layer['output'] > 0)

        #one row per output pixel, matching the rows of layer['cols']
        gradientRows = lossGradient.transpose(0, 2, 3, 1).reshape(-1, filters)

        weightChange = (gradientRows.T @ layer['cols']).reshape(weight.shape)
        biasChange = gradientRows.sum(axis=0)

        #each patch's gradient, then added back onto the pixels it came from ("col2im")
        patchChange = (gradientRows @ weight.reshape(filters, -1)).reshape(batch, outputHeight, outputWidth, channels, filterHeight, filterWidth)
        inputChange = np.zeros(inputShape)
        for i in range(filterHeight):
            for j in range(filterWidth):
                inputChange[:, :, i:i + outputHeight * step:step, j:j + outputWidth * step:step] += patchChange[:, :, :, :, i, j].transpose(0, 3, 1, 2)

        return weightChange, biasChange, inputChange


    """Sends each block's gradient back to the one pixel that was the max (the rest get 0)"""
    def backprop_pool(self, lossGradient, layer):
        size = layer['shape'][0]
        (batch, channels, inputHeight, inputWidth) = layer['input'].shape
        (outputHeight, outputWidth) = (inputHeight // size, inputWidth // size)

        blocks = np.zeros((batch, channels, outputHeight, outputWidth, size * size))
        np.put_along_axis(blocks, layer['argmax'][..., None], lossGradient[..., None], axis=-1)

        #undo the block grouping from pooling()
        blocks = blocks.reshape(batch, channels, outputHeight, outputWidth, size, size).transpose(0, 1, 2, 4, 3, 5)
        inputChange = np.zeros(layer['input'].shape)
        inputChange[:, :, :outputHeight * size, :outputWidth * size] = blocks.reshape(batch, channels, outputHeight * size, outputWidth * size)

        return inputChange


    def train(self, x_train, y_train, x_val, y_val, epochs=10, learningRate=0.005, batchSize=32):

        samples = x_train.shape[0]

        for epoch in range(1, epochs + 1):

            print(f"Epoch {epoch}/{epochs}:")

            # Shuffle the training data at the start of each epoch
            indices = np.random.permutation(samples)

            epoch_loss = 0 #every epoch has its own total loss value
            correct = 0
            for start in range(0, samples, batchSize):
                batch = indices[start:start + batchSize]
                x = prepImages(x_train[batch])
                y = y_train[batch]

                predictions = self.forward(x)
                epoch_loss += cross_entropy_loss(predictions, y) * len(batch)
                correct += np.sum(predictions.argmax(axis=1) == y.argmax(axis=1))
                self.backpropagation(y, predictions, learningRate)

                progress_bar(min(start + batchSize, samples), samples)

            print(f"Loss: {epoch_loss / samples:.4f} - Accuracy: {correct / samples * 100:.2f}%")

            val_accuracy = self.evaluate(x_val, y_val, batchSize)
            print(f"Validation Accuracy after epoch {epoch}: {val_accuracy * 100:.2f}%\n")

        print("Training complete!")


    def predict(self, x, batchSize=256):
        return np.concatenate([self.forward(prepImages(x[i:i + batchSize])) for i in range(0, len(x), batchSize)])


    def evaluate(self, x, y, batchSize=256):
        #argmax = highest number (best guess)
        predictions = self.predict(x, batchSize).argmax(axis=1)

        # % accuracy = correct/all
        return np.mean(predictions == y.argmax(axis=1))


    """Saves every layer's weights to a .npz file"""
    def save(self, path, categories=None):
        arrays = {"fc_weight": self.fcLayer['weight'], "fc_bias": self.fcLayer['bias'],
                  "input_shape": np.array(self.input_shape), "categories": np.array(categories or [])}
        for (i, layer) in enumerate(self.layers):
            arrays[f"layer{i}"] = np.array([layer['type'], layer['shape'][0], layer['shape'][1], layer['step'], layer['filters']])
            if layer['type'] == 'conv':
                arrays[f"layer{i}_weight"] = layer['weight']
                arrays[f"layer{i}_bias"] = layer['bias']
        np.savez(path, **arrays)


    """Loads a network saved with save()
    :returns: (network, categories)
    """
    @staticmethod
    def load(path):
        data = np.load(path)
        layers = list()
        i = 0
        while f"layer{i}" in data:
            (kind, h, w, step, filters) = data[f"layer{i}"]
            layers.append(setLayer(str(kind), (int(h), int(w)), int(step), filters=int(filters)))
            i += 1

        model = Convolutional_NN(tuple(int(n) for n in data["input_shape"]), data["fc_bias"].shape[0], layers)
        for (i, layer) in enumerate(model.layers):
            if layer['type'] == 'conv':
                layer['weight'] = data[f"layer{i}_weight"]
                layer['bias'] = data[f"layer{i}_bias"]
        model.fcLayer['weight'] = data["fc_weight"]
        model.fcLayer['bias'] = data["fc_bias"]

        return (model, [str(c) for c in data["categories"]])


#Helper functions:

#Layer functions: setLayer, shrinkLayer

"""
:param type: "conv" or "pool"
:param shape: (height, width) of the filter or pooling block
:param step: how far the filter moves each time (pool layers need step == size)
:param filters: how many different filters a conv layer learns (each one picks up a different feature)
"""
def setLayer(type, shape, step=1, weight=None, bias=None, filters=1):
        return {
            "type": type,
            "shape": shape,
            "step": step,
            "filters": filters,
            "weight": weight,
            "bias": bias,
        }


def shrinkLayer(layer, shape):

    (height, width) = shape

    if layer['type'] == 'pool':
        return (height // layer['shape'][0], width // layer['shape'][1])

    height = (height - layer["shape"][0]) // layer["step"] + 1
    width = (width - layer["shape"][1]) // layer["step"] + 1

    return (height, width)


"""Turns uint8 images (0-255) into floats between 0 and 1 with a channel axis"""
def prepImages(images):
    images = images.astype(np.float32) / 255
    return images[:, None] if images.ndim == 3 else images


#Activation functions: relu, softmax, cross entropy loss

#relu activation function: sets everything below 0 to 0
def relu(vector):
    return np.maximum(0, vector)


"""Turns each row of scores into probabilities that add up to 1
see here for graph: https://en.wikipedia.org/wiki/Softmax_function
"""
def softmax(scores):
    eV = np.exp(scores - scores.max(axis=1, keepdims=True)) #subtracting the max prevents overflow with large numbers
    return eV / eV.sum(axis=1, keepdims=True)


"""
Cross entropy loss (log loss), averaged over the batch:
loss = -i∑(y[i]*log(p[i]))

where i = each output (letter) it could have guessed
:param y: = the label (1 for "this output is i" 0 for "this output is not i")
:param p: = probability that it would guess the answer "i"
"""
def cross_entropy_loss(p, y):
    p = np.clip(p, 1e-12, 1)  # prevents log(0)
    return -np.sum(y * np.log(p)) / y.shape[0]


def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current >= total else '\r'

    print(f'Progress: [{arrow}{padding}] {current}/{total}, {int(fraction*100)}%', end=ending)
