from tensorflow.keras import models
import numpy as np
from matplotlib import pyplot
from import_data import import_images

image_dir = 'D:/PycharmProjects/image_segmentation/bird_images'
label_dir = 'D:/PycharmProjects/image_segmentation/bird_labels'
image_size = (512, 512)  # images not resized if set to (0,0)
label_size = (128, 128)
images, labels, num_classes, weights = import_images(image_size, label_size, image_dir, label_dir, augment=False)
'''
from PIL import Image
image = Image.open('C:/Users/Sterling/Desktop/images/game2.jpg')
image = np.array(image.resize((512, 512), 0))/ 255.0
images = np.array([image])
print('image size =', np.shape(images))
num_classes = 9
'''
np.random.shuffle(images)

model = models.load_model('./models/seg_512.h5')
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