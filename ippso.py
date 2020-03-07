import numpy as np
from keras import Model
from keras.datasets import mnist
from keras.layers import Conv2D, AveragePooling2D, Dense, MaxPooling2D, Input, Flatten
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from pyswarm import pso

# import pyswarms as ps

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

# _CON_KERNEL_SIZE_NUMBER_OF_BITS = 3
# _CON_FILTER_SIZE_NUMBER_OF_BITS = 7
# _CON_STRIDE_SIZE_NUMBER_OF_BITS = 2
# _POOL_POOL_SIZE_NUMBER_OF_BITS = 2
# _POOL_STRIDE_SIZE_NUMBER_OF_BITS = 2
# _DENSE_NUMBER_OF_BITS = 11

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


def load_data_from_mnist():
    img_shape = (28, 28, 1)
    (x_test, y_test), (x_train, y_train) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], *img_shape)
    x_test = x_test.reshape(x_test.shape[0], *img_shape)
    input_shape = img_shape

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


def is_valid_layer_code(layer_code):
    return 0 <= layer_code[0] <= 255 and 0 <= layer_code[1] <= 255


def build_model(layer_codes):
    input_layer = Input(shape=(28, 28, 1))
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

    output_layer = Dense(10, activation="softmax")(output_layer)
    model = Model(input_layer, output_layer)

    return model


def build_and_fit_model(x, *args):
    x_train, y_train, x_test, y_test = args
    # x = [6, 119, 21, 112, 35, 255, 27, 255]

    layer_codes = np.array(x).reshape(-1, 2).astype(int)
    print(layer_codes)

    try:
        model = build_model(layer_codes)
    except ValueError:
        return 0.0

    model.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=["accuracy"])

    print(model.summary())

    try:
        history = model.fit(x_train, y_train,
                            batch_size=50,
                            epochs=10,
                            verbose=1,
                            validation_data=(x_test, y_test))
    except ValueError:
        return 0.0

    return - history.history["val_acc"][-1]


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
    x_train, y_train, x_test, y_test = load_data_from_mnist()

    for layer_count in range(5, 8):
        xopt, fopt = pso(func=build_and_fit_model,
                         lb=np.zeros(shape=layer_count * 2),
                         ub=np.dstack((np.full(shape=layer_count, fill_value=39),
                                       np.full(shape=layer_count, fill_value=255))).flatten(),
                         minstep=1.0,
                         args=(x_train, y_train, x_test, y_test))
        print(xopt)
        print(fopt)


if __name__ == "__main__":
    main()
    # load_data_from_mnist()
