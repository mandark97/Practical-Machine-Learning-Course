import time

import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical

from data_generator import test_generator, training_generator
from model_wrapper import ModelWrapper, build_config
from utils import save_hist
from models import lenet, changed_lenet, vgg


def reshape_data(a):
    x, y, _ = a
    x = x.reshape(-1, 28, 84, 1)

    y = y.flatten()
    y = to_categorical(y, 256)
    return x, y


def data_flatten(generator, batch_size=32):
    return map(reshape_data, generator(batch_size=batch_size))


BATCH_SIZE = 512
STEPS_PER_EPOCH = 500
EPOCHS = 10
LEARNING_RATE = 'default'
MODEL_NAME = 'lenet_train2'
EARLYSTOP_PATIENCE = 3

callbacks = [
    ModelCheckpoint(filepath='checkpoint.h5', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', min_delta=0,
                  patience=EARLYSTOP_PATIENCE, verbose=2, mode='auto'),
    TensorBoard(log_dir='logs', histogram_freq=0,
                batch_size=BATCH_SIZE, write_grads=True, write_images=True)
]

model = lenet()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

generator = data_flatten(training_generator, batch_size=BATCH_SIZE)
hist = model.fit_generator(generator,
                           steps_per_epoch=STEPS_PER_EPOCH,
                           epochs=EPOCHS,
                           verbose=1,
                           use_multiprocessing=True,
                           workers=-2,
                           callbacks=callbacks,
                           validation_data=generator, validation_steps=30)


score = model.evaluate_generator(data_flatten(test_generator,
                                              batch_size=BATCH_SIZE),
                                 steps=STEPS_PER_EPOCH)
print(score)

config = build_config(MODEL_NAME, LEARNING_RATE, BATCH_SIZE,
                      EPOCHS, STEPS_PER_EPOCH, score[0], score[1])
wrapper = ModelWrapper(model, config=config)
wrapper.save_model()

save_hist(hist)
