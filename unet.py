"""
Original model is located at
    https://colab.research.google.com/drive/1akQBCphVqXw31n5Plykm6Gg7MTVKvfHo

    The definition of the data generators and model and metrics
"""
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import keras.backend as K

class dataGenerator(keras.utils.Sequence):
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        # x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        x = np.zeros((self.batch_size,) + self.img_size+ (3,), dtype="float32")
        x2 = np.zeros(self.img_size+ (3,), dtype="int")


        for j, path in enumerate(batch_input_img_paths):
            # try:
            img = load_img(path, target_size=self.img_size, color_mode="rgb")
            # x[j][:,:,0] = img
            # x2[j] = img
            x[j] = np.array(img)/255

            # import matplotlib.pyplot as plt
            # plt.imshow(x[j])
            # plt.show()
            # except OSError:
                # continue
            # x[j] = x[j]/255.
        y = np.zeros((self.batch_size,) + self.img_size+(3,), dtype="float32")
        for j, path in enumerate(batch_target_img_paths):
            # try:
            img = load_img(path, target_size=self.img_size, color_mode="rgb")
            # y[j] = np.expand_dims(img, 3)
            y[j] = img
            # import matplotlib.pyplot as plt
            # plt.imshow(y[j])
            # plt.show()
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            # y[j] -= 1
            y[j][y[j] > 0] = 1
            # y[j] = y[j] // 255
            # except OSError:
                # continue

        return x, y

"""## Prepare U-Net Xception-style model"""

from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        # x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 1, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    label = 1
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    # y_pred = K.argmax(y_pred, axis=-1)    
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    y_pred = K.expand_dims(y_pred, -1)
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

