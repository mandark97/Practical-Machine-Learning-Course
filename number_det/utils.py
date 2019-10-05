import matplotlib.pyplot as plt
import math
import os

def save_hist(hist):
    plt.figure(figsize=(14, 3))
    plt.subplot(1, 2, 1)
    plt.suptitle('Optimizer : Adam', fontsize=10)
    plt.ylabel('Loss', fontsize=16)
    plt.plot(hist.history['loss'], 'b', label='Training Loss')
    plt.plot(hist.history['val_loss'], 'r', label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(hist.history['acc'], 'b', label='Training Accuracy')
    plt.plot(hist.history['val_acc'], 'r', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('plot.png')


def display_activation(activations, save_location='./'):
    # act_index is the index of the layer eg. 0 is the first layer
    for act_index in range(len(activations)):
        activation = activations[act_index]
        activation_index = 0
        size = activation.shape[-1]
        row_size = math.ceil(math.sqrt(size))
        col_size = math.ceil(math.sqrt(size))
        fig, ax = plt.subplots(
            row_size, col_size, figsize=(row_size*5, col_size*5))
        for row in range(0, row_size):
            for col in range(0, col_size):
                if activation_index < size:
                    ax[row][col].imshow(
                        activation[0, :, :, activation_index], cmap='gray')
                    activation_index += 1
                else:
                    break
        plt.savefig('{0}/activations/activation_layer{1}.png'.format(save_location, act_index))
        plt.clf()

