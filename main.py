import tensorflow as tf
import numpy as np
from import_data import import_images
from get_model import unet_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model_path = './models/seg_512_same.h5'
image_dir = 'D:/PycharmProjects/image_segmentation/bird_images'
label_dir = 'D:/PycharmProjects/image_segmentation/bird_labels'
image_size = (512, 512)
label_size = (512, 512)
def train(images, labels, num_classes, weights):
    print('images shape =', np.shape(images))
    print('labels shape =', np.shape(labels))

    model = unet_model(image_size, num_classes)
    model.summary()

    def dice_loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, 'int32'), num_classes)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        weight_coreection = tf.math.reduce_sum(y_true*tf.constant(weights, dtype=tf.float32), axis=3)
        return loss*weight_coreection

    model.compile(optimizer='adam',
                  loss=dice_loss,
                  metrics=['accuracy'],)

    callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),
                ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_freq='epoch'),]

    model_history = model.fit(
                            x=images,
                            y=labels,
                            epochs=20,
                            batch_size = 10,
                            steps_per_epoch=100,
                            callbacks=callbacks
                            )

    model.save(model_path, include_optimizer=False)

if __name__ == '__main__':
    images, labels, num_classes, weights = import_images(image_size, label_size, image_dir, label_dir, augment=True)
    # visualy_inspect_generated_data(images, labels)
    train(images, labels, num_classes, weights)