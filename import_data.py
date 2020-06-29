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
            label = Image.open(data_dir + '/' + path.split('.')[0] + '_label.jpg', )
            image = np.array(image.resize(image_size, 0))
            label = np.array(label.resize(image_size, 0))
            images.append(image)
            labels.append(label)
        if (i+1)%50 ==0:
            print(int(i/len(paths)*100), '% of images read')
    print(100, '%', "of images read")
    images = np.array(images)/ 255.0
    labels = np.array(labels)

    if augment == True:
        aug_num = 9
        generator = import_data_generator(images, np.expand_dims(labels, axis=3), batch_size=len(images))
        for i, batch in enumerate(generator):
            images = np.concatenate((images, batch[0]))
            labels = np.concatenate((labels, batch[1][:, :, :, 0]))
            print(int((i+1)/aug_num*100), '% of images augmented')
            if i + 1 >= aug_num:
                break
        labels = labels.astype(int)

    labels = np.clip(labels, 0, 8) #have to clip because jpeg sucks. consider different image format
    num_classes = np.amax(labels) + 1

    weights = []
    for i in range(num_classes):
        weights.append((1 - (labels == i).sum()/np.size(labels))*num_classes)
    weights = np.array(weights)

    return images, labels, num_classes, weights

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
    image_size = (224, 224)  # images not resized if set to (0,0)
    num_classes = 9

    import_images(image_size, data_dir)