from data_preprocessing import *
from segnet_models.metrics_and_losses import dice_coef, dice_coef_loss
from segnet_models.segnet import segnet
from segnet_models.modsegnet import modsegnet
from segnet_models.layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from resunet_model import ResUNet
import datetime
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


# Paths
top_path = ''
data_path = top_path+'dataset/'
img_path = data_path+'img/'
mask_path = data_path+'mask/'
red_mask_path = data_path+'mask_red_corr/'
white_mask_path = data_path+'mask_white_corr/'
models_path = top_path+'models/'
train_history_path = top_path+'history_files/'


# Parameters

N_BANDS = 13
N_CLASSES = 2
N_EPOCHS = 20
PATCH_SZ = 96                  # should divide by 16 (160)
BATCH_SIZE = 128
TRAIN_SZ = 16000               # train size 16000/8000
VAL_SZ = 2000                  # validation size 2000/1000
TEST_SZ = 2000                 # validation size 2000/1000



X_DICT, Y_DICT = load_data_and_ndvi(img_path, mask_path=[mask_path, red_mask_path, white_mask_path], sz=PATCH_SZ)
# split into train/validation/test
X_DICT_TRAIN, Y_DICT_TRAIN, X_DICT_VALIDATION, Y_DICT_VALIDATION, X_DICT_TEST, Y_DICT_TEST = split_dataset(X_DICT, Y_DICT, split_rates=[80, 10, 10])

x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
x_test, y_test = get_patches(X_DICT_TEST, Y_DICT_TEST, n_patches=TEST_SZ, sz=PATCH_SZ)

# clean memory
X_DICT, Y_DICT = {}, {}
X_DICT_TRAIN, Y_DICT_TRAIN, X_DICT_VALIDATION, Y_DICT_VALIDATION, X_DICT_TEST, Y_DICT_TEST = {}, {}, {}, {}, {}, {}

IMG_HEIGHT = PATCH_SZ
IMG_WIDTH = PATCH_SZ
IMG_CHANNELS = N_BANDS

for dropout_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:                               # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for depth in [4, 5]:
        for filters in [16, 32]:
            for learning_rate in [0.0001, 0.0005]:
                # SegNet
                model = ResUNet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), filters, depth, dropout_rate)

                opt = tf.keras.optimizers.Adam(learning_rate)
                model.compile(optimizer=opt, loss=dice_coef_loss, metrics=['accuracy', dice_coef])

                timestr = datetime.datetime.now().strftime("(%Y-%m-%d , %H:%M:%S)")
                model_name = model.name + '_{}'.format(timestr)

                checkpoint = ModelCheckpoint(models_path + model_name + '.hdf5', monitor='val_dice_coef', verbose=2,
                                                    save_best_only=True, mode='max', save_weights_only=False)

                # Training
                history = model.fit(x_train, y_train,
                                    batch_size=BATCH_SIZE,
                                    verbose=1,
                                    epochs=N_EPOCHS,
                                    validation_data=(x_val, y_val),
                                    shuffle=False,
                                    callbacks=[checkpoint])

                # results = model.evaluate(x_test, y_test)
                # evaluation
                saved_model = tf.keras.models.load_model(models_path + model_name + '.hdf5', custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})
                results = saved_model.evaluate(x_test, y_test)

                # # Save the model and results
                print(model_name, ' model is saved')

                save_history_to_csv(train_history_path, model_name, history, N_EPOCHS)

                save_resunet_result_to_csv(top_path, model_name, PATCH_SZ, BATCH_SIZE, N_EPOCHS, depth, filters, learning_rate, dropout_rate, history, results)
                

                tf.keras.backend.clear_session()
