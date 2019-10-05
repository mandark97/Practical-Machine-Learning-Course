import numpy as np


class EncoderDecoder(object):

    def __init__(self, chars='0123456789+ ', nr_length=3):
        self.chars = sorted(set(chars))
        self.char_mapping = {c: i for i, c in enumerate(self.chars)}
        self.indices_mapping = {i: c for i, c in enumerate(self.chars)}
        self.nr_length = nr_length

    def _numbers_to_string(self, x_numbers, y_number):
        max_length = self.nr_length * 2 + 1

        X_str = '+'.join([str(n) for n in list(x_numbers)])
        X_str = (' ' * (max_length - len(X_str))) + X_str
        y = str(y_number[0])
        Y_str = (' ' * (self.nr_length - len(y))) + y

        return X_str, Y_str

    def _one_hot_encode(self, number_str):
        x = np.zeros((len(number_str), len(self.chars)))
        for i, c in enumerate(number_str):
            x[i, self.char_mapping[c]] = 1

        return x

    def encode(self, data):
        _, X, Y = data
        x_l = []
        y_l = []
        for (x, y) in zip(X, Y):
            x_str, y_str = self._numbers_to_string(x, y)

            enc_x = self._one_hot_encode(x_str)
            enc_y = self._one_hot_encode(y_str)
            x_l.append(enc_x)
            y_l.append(enc_y)

        return np.array(x_l), np.array(y_l)

    def encode_y(self, data):
        Img, X, Y = data

        y_l = []
        for (x, y) in zip(X, Y):
            _, y_str = self._numbers_to_string(x, y)
            enc_y = self._one_hot_encode(y_str)
            y_l.append(enc_y)

        return Img, np.array(y_l)

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)

        return ''.join(self.indices_mapping[x] for x in x)

    def encode_generator(self, generator, batch_size):
        return map(self.encode, generator(batch_size=batch_size))
