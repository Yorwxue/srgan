import cv2
from scipy import misc
import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter


def all_rescale(path, target_dir):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    itemList = os.listdir(path)
    for item in itemList:
        if item != 'all':
            if os.path.isfile(os.path.join(path, item)):
                try:
                    print(os.path.join(path, item))
                    img = misc.imread(os.path.join(path, item))

                    # gaussian_filter
                    # img = gaussian_filter(img, sigma=0.5)

                    img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
                    misc.imsave(os.path.join(target_dir, item), img)
                except:
                    None
            else:
                all_rescale(os.path.join(path, item), target_dir)


def crop(image, location=None, random_crop=False, image_size=400):
    # location should mark the coordinate of left and up point of crop block(such as location = [0, 0])
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
            image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
        # else:
        #     (h, v) = (0,0)
        #     image = image[(sz1-sz2+v):(sz1+sz2+v), (sz1-sz2+h):(sz1+sz2+h), :]

        if location!=None:
            image = image[location[0]:location[0]+image_size, location[1]:location[1]+image_size, :]

    return image


if __name__ == '__main__':

    img_dir = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/photo_image/large/'
    target_dir = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/photo_image/small/'
    # print('target path: ', target_dir)
    # all_rescale(img_dir, target_dir)

    # --------------------
    image_name = 'IMAG0084.jpg'
    image_path = os.path.join(img_dir, image_name)
    img = misc.imread(image_path)

    processed_img = crop(img, location=[450, 550])
    misc.imsave(os.path.join(target_dir, image_name), processed_img)
