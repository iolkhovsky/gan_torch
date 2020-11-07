import numpy as np


def array_yxc2cyx(arr):
    arr = np.swapaxes(arr, 1, 2)
    arr = np.swapaxes(arr, 0, 1)
    return arr


def array_cyx2yxc(arr):
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr


def normalize_cv_img(cv_img):
    img = cv_img.astype(np.float32)
    img = img - 0.5 * 255.
    img = img * (2. / 255.)
    return img


def denormalize_cv_array(cv_arr):
    img = cv_arr * (255. / 2.)
    img += 0.5 * 255.
    return img


def encode_img(image):
    norm = normalize_cv_img(image)
    return array_yxc2cyx(norm)


def decode_img(tensor):
    arr = array_cyx2yxc(tensor)
    arr = denormalize_cv_array(arr)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

