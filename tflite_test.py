import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from import_data import import_images

image_dir = 'D:/PycharmProjects/image_segmentation/bird_images'
label_dir = 'D:/PycharmProjects/image_segmentation/bird_labels'
image_size = (512, 512)
label_size = (128, 128)
images, labels, num_classes, weights = import_images(image_size, label_size, image_dir, label_dir, augment=True)

np.random.shuffle(images)

tflite_model_path='./models/seg_512_128.tflite'
interpreter = tf.lite.Interpreter(model_path = tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_size = input_details[0]['shape']
print('image_size =', image_size)
print('output_details[0][shape] = ', output_details[0]['shape'][3])

for image in images:
    data = [image.astype('float32')]
    interpreter.set_tensor(input_details[0]['index'], data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.argmax(predictions, axis=3)
    for i in range(num_classes):
        print(np.count_nonzero(predictions == i))
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(predictions[0], vmax=num_classes-1)
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(image)
    pyplot.show()