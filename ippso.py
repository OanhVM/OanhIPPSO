from os.path import join, abspath, pardir

import json_tricks
import numpy as np
from keras import Model
from keras.datasets import mnist
from keras.layers import Conv2D, AveragePooling2D, Dense, MaxPooling2D, Input, Flatten
from keras.optimizers import Adadelta
from keras.utils.layer_utils import count_params
from keras_preprocessing.image import ImageDataGenerator
from pyswarm import pso

_CON_LAYER_STRIDE_END_BITS = 15
_CON_LAYER_STRIDE_START_BITS = 14
_CON_LAYER_FILTER_END_BITS = 13
_CON_LAYER_FILTER_START_BITS = 7
_CON_LAYER_KERNEL_END_BITS = 6
_CON_LAYER_KERNEL_START_BITS = 4

_POOL_LAYER_TYPE_BITS = 9
_POOL_LAYER_STRIDE_END_BITS = 8
_POOL_LAYER_STRIDE_START_BITS = 7
_POOL_LAYER_SIZE_END_BITS = 6
_POOL_LAYER_SIZE_START_BITS = 5

_FULL_LAYER_END_BITS = 15
_FULL_LAYER_START_BITS = 5

_CON_LAYER_RANGE_1 = 0
_CON_LAYER_RANGE_2 = 15
_CONV_RANGE = (0, 15)
_POOL_LAYER_RANGE_1 = 16
_POOL_LAYER_RANGE_2 = 23
_FULL_LAYER_RANGE_1 = 24
_FULL_LAYER_RANGE_2 = 31
_DISABLED_LAYER_RANGE_1 = 32
_DISABLED_LAYER_RANGE_2 = 39
_START_RANGE = 0
_END_RANGE = 255

_IP_NUMBER_OF_BITS = 16

_RESULTS_FILE_NAME = "results.txt"


def load_data_from_mnist():
    img_shape = (28, 28, 1)
    (x_test, y_test), (x_train, y_train) = mnist.load_data()
    # data = ImagesetLoader.load("mrd")
    # # training images
    # x_train = data.train["images"]
    # # training labels
    # y_train = data.train["labels"]
    # # test images
    # x_test = data.test["images"]
    # # test labels
    # y_test = data.test["labels"]

    x_train = x_train.reshape(x_train.shape[0], *img_shape)
    x_test = x_test.reshape(x_test.shape[0], *img_shape)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    return x_train, y_train, x_test, y_test


_DATA_SET_PATH = join(abspath(join(abspath(__file__), pardir)), "data", "intel")
_TRAIN_DIR = join(_DATA_SET_PATH, "train")
_VALIDATION_DIR = join(_DATA_SET_PATH, "test")

_BATCH_SIZE = 500

_IMG_SHAPE = (150, 150, 3)


def generate_data():
    train_image_generator = ImageDataGenerator(rotation_range=45,
                                               width_shift_range=.15,
                                               height_shift_range=.15,
                                               horizontal_flip=True,
                                               zoom_range=0.5)

    train_data_gen = train_image_generator.flow_from_directory(batch_size=_BATCH_SIZE,
                                                               directory=_TRAIN_DIR,
                                                               shuffle=True,
                                                               target_size=_IMG_SHAPE[:2],
                                                               class_mode="categorical")

    test_image_generator = ImageDataGenerator()

    val_data_gen = test_image_generator.flow_from_directory(batch_size=_BATCH_SIZE,
                                                            directory=_VALIDATION_DIR,
                                                            shuffle=True,
                                                            target_size=_IMG_SHAPE[:2],
                                                            class_mode="categorical")
    # MUAHAHAHAHAHAHAHA!!!!!
    temp = train_data_gen
    train_data_gen = val_data_gen
    val_data_gen = temp

    return train_data_gen, val_data_gen


def is_valid_layer_code(layer_code):
    return 0 <= layer_code[0] <= 255 and 0 <= layer_code[1] <= 255


def build_model(layer_codes):
    input_layer = Input(shape=_IMG_SHAPE)
    output_layer = input_layer

    for idx, layer_code in enumerate(layer_codes):
        if is_valid_layer_code(layer_code):

            bits = dec_to_bin_8bits(layer_code[0]) + dec_to_bin_8bits(layer_code[1])

            if _CONV_RANGE[0] <= layer_code[0] <= _CONV_RANGE[1]:
                print("{}.{} belong to {}".format(*layer_code, "CONV_LAYER"))
                try:
                    conv_layer = decode_cov_layer(bits)
                except ValueError:
                    continue
                output_layer = conv_layer(output_layer)

            elif _POOL_LAYER_RANGE_1 <= layer_code[0] <= _POOL_LAYER_RANGE_2:
                if idx == 0:
                    raise ValueError("POOL_LAYER must not be at the beginning.")
                else:
                    print("{}.{} belong to {}".format(*layer_code, "POOL_LAYER"))
                    pooling_layer = decode_pooling_layer(bits)
                    output_layer = pooling_layer(output_layer)

            elif _FULL_LAYER_RANGE_1 <= layer_code[0] <= _FULL_LAYER_RANGE_2:
                print("{}.{} belong to {}".format(*layer_code, "DENSE_LAYER"))
                if idx != len(layer_codes) - 1:
                    raise ValueError("DENSE_LAYER must be at the end.")
                else:
                    output_layer = Flatten()(output_layer)

                    dense_layer = decode_dense(bits)
                    output_layer = dense_layer(output_layer)

    output_layer = Dense(6, activation="softmax")(output_layer)
    model = Model(input_layer, output_layer)

    return model


def build_and_fit_model(x, *args):
    train_data_gen, val_data_gen = args

    layer_codes = np.array(x).reshape(-1, 2).astype(int)
    print(layer_codes)

    try:
        model = build_model(layer_codes)
    except ValueError:
        return 0.0

    model.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=["accuracy"])

    print(model.summary())
    trainable_count = count_params(model.trainable_weights)
    print("trainable_count", trainable_count)
    if trainable_count > 0:
        try:
            history = model.fit_generator(
                generator=train_data_gen,
                steps_per_epoch=train_data_gen.n // train_data_gen.batch_size,
                epochs=10,
                validation_data=val_data_gen,
                validation_steps=val_data_gen.n // val_data_gen.batch_size,
                verbose=1
            )
        except ValueError:
            return 0.0

        val_acc = history.history["val_acc"][-1]
        with open(_RESULTS_FILE_NAME, "a") as f:
            f.write("{}\n".format(
                json_tricks.dumps({
                    "model_config": model.to_json(),
                    "val_acc": val_acc,
                }, indent=4, sort_keys=True)
            ))

    else:
        val_acc = 0.0

    return - val_acc


def decode_cov_layer(bits):
    kernel_size = bin_to_dec_plus(str(bits[_CON_LAYER_KERNEL_START_BITS:_CON_LAYER_KERNEL_END_BITS + 1]))
    filter_size = bin_to_dec_plus(str(bits[_CON_LAYER_FILTER_START_BITS:_CON_LAYER_FILTER_END_BITS + 1]))
    strides = bin_to_dec_plus(str(bits[_CON_LAYER_STRIDE_START_BITS:_CON_LAYER_STRIDE_END_BITS + 1]))

    return Conv2D(filters=filter_size,
                  kernel_size=kernel_size,
                  strides=strides,
                  activation="relu")


def decode_pooling_layer(bits):
    pool_size = bin_to_dec_plus(str(bits[_POOL_LAYER_SIZE_START_BITS:_POOL_LAYER_SIZE_END_BITS + 1]))
    strides = bin_to_dec_plus(str(bits[_POOL_LAYER_STRIDE_START_BITS:_POOL_LAYER_STRIDE_END_BITS + 1]))

    pool_type = bin_to_dec_plus(bits[_POOL_LAYER_TYPE_BITS])

    if pool_type == 1:
        return MaxPooling2D(pool_size=pool_size, strides=strides)
    else:
        return AveragePooling2D(pool_size=pool_size, strides=strides)


def decode_dense(bits):
    units = bin_to_dec_plus(str(bits[_FULL_LAYER_START_BITS:_FULL_LAYER_END_BITS + 1]))

    return Dense(units=units, activation="relu")


def dec_to_bin_8bits(n):
    return format(n, "08b")


def bin_to_dec_plus(n):
    return int(str(n), 2) + 1


def main():
    # x_train, y_train, x_test, y_test = load_data_from_mnist()
    train_data_gen, val_data_gen = generate_data()

    for layer_count in range(5, 6):
        xopt, fopt = pso(func=build_and_fit_model,
                         lb=np.zeros(shape=layer_count * 2),
                         ub=np.dstack((np.full(shape=layer_count, fill_value=39),
                                       np.full(shape=layer_count, fill_value=255))).flatten(),
                         # minstep=1.0,
                         maxiter=50,
                         args=(train_data_gen, val_data_gen))
        print(xopt)
        print(fopt)


if __name__ == "__main__":
    main()
