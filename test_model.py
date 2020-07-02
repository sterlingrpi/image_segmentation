from tensorflow.keras import models
import numpy as np
from matplotlib import pyplot
from import_data import import_images

image_size = (256, 256)
data_dir = 'D:/PycharmProjects/ssd-tf2/dataset/bird_images'
images, labels, num_classes, weights = import_images(image_size, data_dir, augment=False)

np.random.shuffle(images)

model = models.load_model('model.h5')
for image in images:
    predictions = model.predict(np.array([image]))
    predictions = np.argmax(predictions, axis=3)
    for i in range(num_classes):
        print(np.count_nonzero(predictions == i))
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(predictions[0], vmax=num_classes-1)
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(image)
    pyplot.show()