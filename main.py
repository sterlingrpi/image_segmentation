import tensorflow as tf
import numpy as np
from import_data import import_images
from get_model import unet_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

image_size = (256, 256)
data_dir = 'D:/PycharmProjects/ssd-tf2/dataset/bird_images'
images, labels, num_classes, weights = import_images(image_size, data_dir, augment=True)
print('images shape =', np.shape(images))
print('labels shape =', np.shape(labels))

#visualy_inspect_generated_data(images, labels)

TRAIN_LENGTH = int(len(images)*0.8)
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

model = unet_model(image_size, num_classes)
model.summary()

def weighted_categorical_crossentropy(weights, num_classes):
    def wcce(y_true, y_pred):
        print('y_true =', y_true)
        print('y_pred =', y_pred)
        print('y_true one hot =', tf.one_hot(y_true, num_classes))
        return  tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return wcce

def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.keras.backend.flatten(tf.keras.backend.one_hot(tf.keras.backend.cast(y_true, 'int32'), num_classes=num_classes)[...,1:])
    y_pred_f = tf.keras.backend.flatten(y_pred[...,1:])
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f, axis=-1)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f, axis=-1)
    return tf.keras.backend.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat_loss(y_true, y_pred):
    return 1 - dice_coef_9cat(y_true, y_pred)

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta = 0.5

    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = tf.keras.backend.sum(p0 * g0, (0, 1, 2))
    den = num + alpha * tf.keras.backend.sum(p0 * g1, (0, 1, 2)) + beta * tf.keras.backend.sum(p1 * g0, (0, 1, 2))
    T = tf.keras.backend.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]
    Ncl = tf.cast(tf.shape(y_true)[-1], 'float32')
    return Ncl - T

def dice_loss(y_true, y_pred):
    y_true = tf.one_hot(tf.cast(y_true, 'int32'), num_classes)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
    weight_tensor = tf.constant(weights, dtype=tf.float32)
    y_true_x_weight_tensor = y_true*weight_tensor
    weight_coreection = tf.math.reduce_sum(y_true_x_weight_tensor, axis=3)
    return loss*weight_coreection

model.compile(optimizer='adam',
              loss=dice_loss,
              metrics=['accuracy'],)

VAL_SUBSPLITS = 5
#VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, verbose=0),
    ModelCheckpoint('model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'),
]

model_history = model.fit(
                        x=images,
                        y=labels,
                        epochs=20,
                        batch_size = 25,
                        steps_per_epoch=25,
                        #class_weight = class_weights,
                        #sample_weight = sample_weights,
                        #validation_steps=VALIDATION_STEPS,
                        #validation_data=test_dataset,
                        callbacks=callbacks
                        )

model.save('model.h5', include_optimizer=False)

'''
model.summary()

result = model(images)
for i in range(9):
    plt.imshow(result[0,:,:,i])
    plt.show()
'''