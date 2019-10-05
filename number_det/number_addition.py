import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from data_generator import training_generator, test_generator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from encoder_decoder import EncoderDecoder
from model_wrapper import ModelWrapper, build_config

CHARS = '0123456789+ '
enc_dec = EncoderDecoder(CHARS)


def encode_generator(generator, batch_size):
    return map(enc_dec.encode, generator(batch_size=batch_size))


MODEL_NAME = 'number_addition'
LEARNING_RATE = 'default'
BATCH_SIZE = 512
STEPS_PER_EPOCH = 500
EPOCHS = 10
HIDDEN_SIZE = 256
RNN = layers.LSTM

callbacks = [
    ModelCheckpoint(filepath='checkpoint.h5',
                    verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', min_delta=0,
                  patience=10, verbose=2, mode='auto'),
    TensorBoard(log_dir='logs', histogram_freq=0,
                batch_size=BATCH_SIZE, write_grads=True, write_images=True)
]
model = Sequential([
    layers.InputLayer((7, len(CHARS))),
    RNN(HIDDEN_SIZE),
    layers.RepeatVector(3),
    RNN(128, return_sequences=True),
    layers.TimeDistributed(layers.Dense(len(CHARS), activation='softmax'))
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

train_generator = encode_generator(training_generator, BATCH_SIZE)

hist = model.fit_generator(train_generator,
                           steps_per_epoch=STEPS_PER_EPOCH,
                           epochs=EPOCHS,
                           verbose=1,
                           use_multiprocessing=True,
                           workers=-2,
                           callbacks=callbacks,
                           validation_data=train_generator, validation_steps=30)

score = model.evaluate_generator(encode_generator(
    test_generator, BATCH_SIZE), steps=STEPS_PER_EPOCH)
print(score)

config = build_config(MODEL_NAME, LEARNING_RATE, BATCH_SIZE,
                      EPOCHS, STEPS_PER_EPOCH, score[0], score[1])
wrapper = ModelWrapper(model, config=config)
wrapper.save_model()
