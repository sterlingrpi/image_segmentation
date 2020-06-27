import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from import_data import import_images

image_size = (256, 256)
data_dir = 'D:/PycharmProjects/ssd-tf2/dataset/bird_images'
images, labels, num_classes, weights = import_images(image_size, data_dir, augment=False)

np.random.shuffle(images)

tflite_model_path='./seg.tflite'
interpreter = tf.lite.Interpreter(model_path = tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
for image in images:
    data = [image.astype('float32')]
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.argmax(predictions, axis=3)
    for i in range(num_classes):
        print(np.count_nonzero(predictions == i))
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(predictions[0])
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(image)
    pyplot.show()