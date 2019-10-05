import csv
import os
import json
from tensorflow.keras.models import load_model, model_from_json, save_model


class ModelWrapper(object):
    def __init__(self, model, config={}):
        self.model = model
        self.model_name = config['model_name']
        self.config = config

    def save_model(self):
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)

        save_model(self.model, '{}/model.h5'.format(self.model_name))

        with open('{}/model.json'.format(self.model_name), 'w') as file:
            file.write(self.model.to_json())

        with open('{}/summary.txt'.format(self.model_name), 'w') as file:
            self.model.summary(print_fn=lambda x: file.write(x + '\n'))
        self._save_to_csv()

    @staticmethod
    def load_model(model_name, with_weights=True):
        with open('{}/model.json'.format(model_name), 'r') as json_file:
            architecture = json.load(json_file)
        model = model_from_json(json.dumps(architecture))

        if with_weights:
            model.load_weights('{}/model.h5'.format(model_name))
        return model

    def _save_to_csv(self):
        csv_name = 'results.csv'

        with open(csv_name, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                                    "model_name", "learning_rate", "batch_size", "epochs", "steps_per_epoch", "score", "accuracy", "extra"])
            writer.writerow(self.config)


def build_config(model_name, learning_rate, batch_size, epochs, steps_per_epoch, score, accuracy, extra={}):
    return {
        'model_name': model_name,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'score': score,
        'accuracy': accuracy,
        'extra': extra
    }
