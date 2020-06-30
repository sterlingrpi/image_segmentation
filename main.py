import tensorflow as tf
import numpy as np
from import_data import import_images
from get_model import unet_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

image_size = (256, 256)
data_dir = 'D:/PycharmProjects/ssd-tf2/dataset/bird_images'
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
                ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'),]

    model_history = model.fit(
                            x=images,
                            y=labels,
                            epochs=20,
                            batch_size = 25,
                            steps_per_epoch=25,
                            callbacks=callbacks
                            )

    model.save('model.h5', include_optimizer=False)

if __name__ == '__main__':
    images, labels, num_classes, weights = import_images(image_size, data_dir, augment=True)
    # visualy_inspect_generated_data(images, labels)
    train(images, labels, num_classes, weights)