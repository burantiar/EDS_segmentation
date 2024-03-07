import os
import numpy as np
import random
import rasterio as rio
import tifffile as tiff
import csv



def calc_min_max(data):
    """
    Function to calculate min and max of the dataset for each band separately
    """
    min_list = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1]
    max_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]

    for band in range(13):
        for key in data.keys():
            temp_min = np.min(data[key][:, :, band])
            temp_max = np.max(data[key][:, :, band])
            if temp_min < min_list[band]:
                min_list[band] = temp_min
            if temp_max > max_list[band]:
                max_list[band] = temp_max

    return min_list, max_list


def normalize_data(data):
    min_list, max_list = calc_min_max(data)

    for key in data.keys():
        for band in range(13):
            data[key][:, :, band] = (data[key][:, :, band].astype(np.float32) - min_list[band]) / (max_list[band] - min_list[band])
    print('Data normalized')
    return data


def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    # print('img.shape',img.shape,'img.shape[0]',img.shape[0],'img.shape[1]',img.shape[1],'mask.shape',mask.shape)  

    # assert len(img.shape) == 3 and img.shape[0] >= sz and img.shape[1] >= sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz), :]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz), :]

    return patch_img, patch_mask


def get_patches(x_dict, y_dict, n_patches, sz=160):
    x = list()
    y = list()
    total_patches = 0
    random.seed(42)
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        # print(img_id)
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask[:, :, :], sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)


def load_data_and_ndvi(img_path, mask_path, sz):
    # set masks paths
    new_mask_path, red_path, white_path = mask_path[0], mask_path[1], mask_path[2]

    # get all images names
    trainIds = []
    for root, dirs, files in os.walk(img_path):
        for name in files:
            if os.path.join(root, name).endswith('.tif'):
                trainIds.append(str(name[:-4]))

    dataset_length = len(trainIds)
    print(dataset_length)

    # shaffle image names
    random.Random(42).shuffle(trainIds)

    X_DICT = {}
    Y_DICT = {}

    print('Reading images')
    for img_num, img_name in enumerate(trainIds):
        image = tiff.imread(img_path + '{}.tif'.format(img_name))
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        assert not np.any(np.isnan(image)) or np.all(np.isfinite(image))
        # exclude image if size < patch size
        if image.shape[0] < sz or image.shape[1] < sz or image.shape[2] != 12:
            print(f'{img_name} is excluded, shape is less than {sz}px')
            continue

        # Calculate NDVI
        ndvi = (image[:, :, 7].astype(float) - image[:, :, 3].astype(float)) / (image[:, :, 7] + image[:, :, 3])

        # Add a new band
        ndvi = ndvi.reshape(ndvi.shape[0], ndvi.shape[1], 1)
        image = np.concatenate((image, ndvi), axis=2)

        # check if masks exist
        ismask_red = os.path.isfile(red_path+'mask_'+img_name+'_Red.png')
        ismask_white = os.path.isfile(white_path+'mask_'+img_name+'_White.png')
        ismask = os.path.isfile(new_mask_path+'mask_'+img_name+'_ALL.png')

        # both red and white
        if ismask_red is True and ismask_white is True and ismask is False:
            with rio.open(red_path + '{}.png'.format('mask_' + img_name + '_Red')) as img:
                mask_red = img.read().transpose([1, 2, 0]) / 1.0
            with rio.open(white_path + '{}.png'.format('mask_' + img_name + '_White')) as img:
                mask_white = img.read().transpose([1, 2, 0]) / 1.0
            mask_red, mask_white = np.array(mask_red), np.array(mask_white)
            mask = mask_red + mask_white
            # if there is overlapping, set value = 1. (= vineyard)
            if 2 in np.unique(mask):
                mask[np.where(mask == 2.)] = 1.
        # only red
        elif ismask_red is True and ismask_white is False and ismask is False:
            with rio.open(red_path + '{}.png'.format('mask_' + img_name + '_Red')) as img:
                mask_red = img.read().transpose([1, 2, 0]) / 1.0
            mask = np.array(mask_red)
        # only white
        elif ismask_red is False and ismask_white is True and ismask is False:
            with rio.open(white_path + '{}.png'.format('mask_' + img_name + '_White')) as img:
                mask_white = img.read().transpose([1, 2, 0]) / 1.0
            mask = np.array(mask_white)
        elif ismask_red is False and ismask_white is False and ismask is True:
            with rio.open(new_mask_path + '{}.png'.format('mask_' + img_name + '_ALL')) as img:
                mask = img.read().transpose([1, 2, 0]) / 1.0
            mask = np.array(mask)

        # resize mask
        if image.shape[0:2] != mask.shape[0:2]:
            print("Width and height don't match! Image {} shape is {} and mask shape is {}".format(img_name, image.shape[0:2], mask.shape[0:2]))
            continue

        X_DICT[img_num] = image[:, :, :]
        Y_DICT[img_num] = mask[:, :, :]

    # normalize images
    X_DICT = normalize_data(X_DICT)
    print('images are normalized')

    return X_DICT, Y_DICT


def split_dataset(images, masks, split_rates=[80, 10, 10]):
    print('splitting...')
    X_DICT_TRAIN = {}
    Y_DICT_TRAIN = {}
    X_DICT_VALIDATION = {}
    Y_DICT_VALIDATION = {}
    X_DICT_TEST = {}
    Y_DICT_TEST = {}

    dataset_length = len(images)

    for img_num, img_key in enumerate(list(images.keys())):
        # img_id = str(img_num).zfill(3)

        if img_num < split_rates[0]/100 * dataset_length:
            X_DICT_TRAIN[img_key] = images[img_key][:, :, :]
            Y_DICT_TRAIN[img_key] = masks[img_key][:, :, :]
        elif img_num >= split_rates[0]/100 * dataset_length and img_num < (split_rates[0]+split_rates[1])/100 * dataset_length:
            X_DICT_VALIDATION[img_key] = images[img_key][:, :, :]
            Y_DICT_VALIDATION[img_key] = masks[img_key][:, :, :]
        elif img_num >= (split_rates[0]+split_rates[1])/100 * dataset_length:
            # print(img_name)
            X_DICT_TEST[img_key] = images[img_key][:, :, :]
            Y_DICT_TEST[img_key] = masks[img_key][:, :, :]

    print('dataset is split')

    print("TRAIN: ", len(X_DICT_TRAIN), len(Y_DICT_TRAIN))
    print("VAL: ", len(X_DICT_VALIDATION), len(Y_DICT_VALIDATION))
    print("TEST: ", len(X_DICT_TEST), len(Y_DICT_TEST))

    return X_DICT_TRAIN, Y_DICT_TRAIN, X_DICT_VALIDATION, Y_DICT_VALIDATION, X_DICT_TEST, Y_DICT_TEST


def save_history_to_csv(path, model_name, history, n_epochs):
    header = list(history.history.keys())
    header.insert(0, 'epoch')
    epochs = [e for e in range(1, n_epochs+1)]
    row_list = zip(epochs, history.history['loss'], history.history['accuracy'], history.history['dice_coef'],
                   history.history['val_loss'], history.history['val_accuracy'], history.history['val_dice_coef'])

    with open(path + str(model_name) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(row_list)


def save_result_to_csv(path, model_name, patch_size, batch_size, n_epochs, filters, learning_rate, history, results):
    with open(path + 'training_results.csv', 'a') as csvfile:
        fieldnames = ['Model', 'Input_size', 'Batch_size', 'Epochs', 'Filters', 'Learning_rate', 'Train_loss', 'Train_acc', 'Train_dice_coef',
                      'Val_loss', 'Val_acc', 'Val_dice_coef', 'Test_loss', 'Test_acc', 'Test_dice_coef']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Model': str(model_name),
                         'Input_size': patch_size,
                         'Batch_size': batch_size,
                         'Epochs': n_epochs,
                         'Filters': filters,
                         'Learning_rate': learning_rate,
                         'Train_loss': "{0:.4f}".format(history.history['loss'][-1]),
                         'Train_acc': "{0:.2f}".format(history.history['accuracy'][-1]),
                         'Train_dice_coef': "{0:.2f}".format(history.history['dice_coef'][-1]),
                         'Val_loss': "{0:.4f}".format(history.history['val_loss'][-1]),
                         'Val_acc': "{0:.2f}".format(history.history['val_accuracy'][-1]),
                         'Val_dice_coef': "{0:.2f}".format(history.history['val_dice_coef'][-1]),
                         'Test_loss': "{0:.4f}".format(results[0]),
                         'Test_acc': "{0:.2f}".format(results[1]),
                         'Test_dice_coef': "{0:.2f}".format(results[2])
                         })
        

def save_resunet_result_to_csv(path, model_name, patch_size, batch_size, n_epochs, depth, filters, learning_rate, dropout_rate, history, results):
    with open(path + 'training_results_resunet.csv', 'a') as csvfile:
        fieldnames = ['Model', 'Input_size', 'Batch_size', 'Epochs', 'Depth', 'Filters', 'Learning_rate', 'Dropout_rate', 'Train_loss', 'Train_acc', 'Train_dice_coef',
                      'Val_loss', 'Val_acc', 'Val_dice_coef', 'Test_loss', 'Test_acc', 'Test_dice_coef']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Model': str(model_name),
                         'Input_size': patch_size,
                         'Batch_size': batch_size,
                         'Epochs': n_epochs,
                         'Depth': depth,
                         'Filters': filters,
                         'Learning_rate': learning_rate,
                         'Dropout_rate': dropout_rate,
                         'Train_loss': "{0:.4f}".format(history.history['loss'][-1]),
                         'Train_acc': "{0:.2f}".format(history.history['accuracy'][-1]),
                         'Train_dice_coef': "{0:.2f}".format(history.history['dice_coef'][-1]),
                         'Val_loss': "{0:.4f}".format(history.history['val_loss'][-1]),
                         'Val_acc': "{0:.2f}".format(history.history['val_accuracy'][-1]),
                         'Val_dice_coef': "{0:.2f}".format(history.history['val_dice_coef'][-1]),
                         'Test_loss': "{0:.4f}".format(results[0]),
                         'Test_acc': "{0:.2f}".format(results[1]),
                         'Test_dice_coef': "{0:.2f}".format(results[2])
                         })
