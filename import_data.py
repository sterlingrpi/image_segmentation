import os
import numpy as np
from PIL import Image
import tensorflow as tf

def import_images(image_size, data_dir, augment):
    paths = os.listdir(data_dir)
    images = []
    labels = []
    for i, path in enumerate(paths):
        if 'label' not in path:
            image = Image.open(data_dir + '/' + path)
            label = Image.open(data_dir + '/' + path.split('.')[0] + '_label.jpg')
            image = image.resize(image_size, 0)
            label = label.resize(image_size, 0)
            images.append(image)
            labels.append(label)
        if (i+1)%50 ==0:
            print(int(i/len(paths)*100), '% of images read')
    print(100, '%', "of images read")

    if augment == True:
        rotate_range = 10
        translate_range = 20
        aug_num = 9
        og_num_images = len(images)
        for i in range(aug_num):
            for j in range(og_num_images):
                aug_image, aug_label = augment_image(images[j], labels[j], rotate_range, translate_range)
                images.append(aug_image)
                labels.append(aug_label)
            print(int((i + 1) / aug_num * 100), '% of images augmented')

    image_array = np.ndarray(shape=(len(images), image_size[0], image_size[1], 3), dtype=np.float32)
    for i, image in enumerate(images):
        image_array[i] = np.array(image)
    label_array = np.ndarray(shape=(len(labels), image_size[0], image_size[1]), dtype=np.float32)
    for i, label in enumerate(labels):
        label_array[i] = np.array(label)
    image_array = image_array / 255.0
    label_array = label_array.astype(int)

    num_classes = np.amax(label_array) + 1
    label_array = np.clip(label_array, 0, 8) #have to clip because jpeg sucks. consider different image format

    weights = []
    for i in range(num_classes):
        weights.append((1 - (label_array == i).sum()/np.size(label_array))*num_classes)
    weights = np.array(weights)

    return image_array, label_array, num_classes, weights

def augment_image(image, label, rotate_range, translate_range):
    degrees = np.random.randint(-rotate_range, rotate_range)
    image = image.rotate(degrees)
    label = label.rotate(degrees)

    method = Image.EXTENT
    a = np.random.randint(-translate_range, translate_range)/100
    b = np.random.randint(-translate_range, translate_range)/100
    c = np.random.randint(100 - translate_range, 100 + translate_range)/100
    d = np.random.randint(100 - translate_range, 100 + translate_range)/100
    data_image = (image.size[0]*a, image.size[1]*b, image.size[0]*c, image.size[1]*d)
    data_label = (label.size[0]*a, label.size[1]*b, label.size[0]*c, label.size[1]*d)
    image = image.transform(size=image.size, method=method, data=data_image, resample=0, fill=1, fillcolor=None)
    label = label.transform(size=label.size, method=method, data=data_label, resample=0, fill=1, fillcolor=None)
    return image, label

def import_data_generator(images, labels, batch_size):
    seed = 1864
    rotation_range = 10
    width_shift_range = 0.2
    height_shift_range = 0.2
    shear_range = 0.2
    zoom_range = 0.2
    horizontal_flip = True
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=rotation_range,
                                                                    width_shift_range=width_shift_range,
                                                                    height_shift_range=height_shift_range,
                                                                    shear_range=shear_range,
                                                                    zoom_range=zoom_range,
                                                                    horizontal_flip=horizontal_flip,
                                                                    fill_mode='nearest')
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=rotation_range,
                                                                    width_shift_range=width_shift_range,
                                                                    height_shift_range=height_shift_range,
                                                                    shear_range=shear_range,
                                                                    zoom_range=zoom_range,
                                                                    horizontal_flip=horizontal_flip,
                                                                    fill_mode='constant')

    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(labels, augment=True, seed=seed)

    image_generator = image_datagen.flow(
                        images,
                        seed=seed,
                        batch_size=batch_size)
    mask_generator = mask_datagen.flow(
                        labels,
                        seed=seed,
                        batch_size=batch_size)

    train_generator = zip(image_generator, mask_generator)

    return train_generator

def import_data_generator_from_directory(image_path, label_path, input_size, output_size, batch_size):
    seed = 1864
    data_gen_args = dict(rotation_range=5,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=1)
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(data_gen_args)

    image_generator = image_datagen.flow_from_directory(
        image_path,
        target_size=input_size,
        seed=seed,
        batch_size=batch_size)
    mask_generator = mask_datagen.flow_from_directory(
        label_path,
        target_size=output_size,
        seed=seed,
        batch_size=batch_size)
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    return train_generator

if __name__ == '__main__':
    data_dir = 'D:/PycharmProjects/ssd-tf2/dataset/bird_images'
    image_size = (256, 256)  # images not resized if set to (0,0)

    images, labels, num_classes, weights = import_images(image_size, data_dir, augment=True)