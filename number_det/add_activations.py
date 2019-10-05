from tensorflow.keras import Model
import os
from utils import display_activation
from model_wrapper import ModelWrapper
from data_generator import test_generator


def reshape_data(a):
    x, _, _ = a
    x = x.reshape(-1, 28, 84, 1)
    return x


def data_flatten(generator, batch_size=32):
    return map(reshape_data, generator(batch_size=batch_size))


X = next(data_flatten(test_generator, 512))

for model_dir in os.listdir('classification_model_results'):
    try:
        model = ModelWrapper.load_model(
            'classification_model_results/{}'.format(model_dir))
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(X, verbose=1)
        display_activation(activations,
                            'classification_model_results/{}'.format(model_dir))
    except:
        print(model_dir)
