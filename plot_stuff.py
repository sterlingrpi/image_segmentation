import matplotlib.pyplot as plt
import tensorflow as tf

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def visualy_inspect_generated_data(train_images, train_labels):
    plt.figure(figsize=(10, 10))
    plt.suptitle('Images with training labels', fontsize=16)
    for i in range(10):
        j = 2 * i
        plt.subplot(5, 5, j + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i, :, :, :], cmap=plt.cm.binary)
        plt.subplot(5, 5, j + 2)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_labels[i, :, :, 0], cmap=plt.cm.binary)
    plt.show()