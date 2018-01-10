import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from scipy import misc


def noisy(noise_typ, image):
    if noise_typ == "speckle":
        alot = 2 * image.max() * np.random.random(image.shape)
        noisy = image + alot
        return noisy

    elif noise_typ == "spot":
        noisy = image.copy()
        row, col, ch = image.shape

        radio_of_max_nb_of_noisy = 0.000025
        max_radius = np.min([np.int(row * 0.02), np.int(col * 0.02)])

        nb_speckle = np.int(np.random.normal((row*col*ch)*radio_of_max_nb_of_noisy))

        print('add %d speckle' % nb_speckle)
        locat_x_of_speckle = np.random.randint(0, row, nb_speckle)
        locat_y_of_speckle = np.random.randint(0, col, nb_speckle)

        for each in range(nb_speckle):
            noisy_level = np.random.rand()
            radius = np.random.randint(0, max_radius)
            for i in range(radius*2):
                x = locat_x_of_speckle - radius + i
                if x >= row:
                    continue
                for j in range(radius*2):
                    y = locat_y_of_speckle - radius + j
                    if y >= col:
                        continue
                    if np.power(((x-locat_x_of_speckle) ** 2 + (y-locat_y_of_speckle) ** 2), 1/2) <= radius:
                        for c in range(3):
                            noisy[x, y, c] = noisy[x, y, c] * noisy_level
        return noisy


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')


src_path, _ = os.path.split(os.path.realpath(__file__))
# img_path = os.path.join(src_path, '991px-Panzer_IV_Wreck_Normandy.jpg')
img_path = os.path.join(src_path, 'F-14A_VF-84_at_NAS_Fallon_1988.JPEG')

img = cv2.imread(img_path)

# add noise
# ------------------------
noisy_image = img.copy()
noisy_image0 = noisy('speckle', noisy_image)
noisy_image = noisy_image0.copy()
# # Gaussian Filtering
# noisy_image = cv2.GaussianBlur(noisy_image, (5, 5), 0)
# # Median Filtering
# noisy_image = cv2.medianBlur(noisy_image, 5)
# # Bilateral Filtering
# noisy_image = cv2.bilateralFilter(noisy_image, 9, 75, 75)
# ------------------------

# Normalize image to between 0 and 255
noisy_image *= 255.0/noisy_image.max()
noisy_image = noisy_image.astype(np.uint8)

# resize
noisy_image = cv2.resize(noisy_image, None, fx=0.25, fy=0.25)
# noisy_image = cv2.resize(noisy_image, (0, 0), fx=4, fy=4)
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

# show result
cv2.imshow('original image', img)
cv2.imshow('processed image', noisy_image)
cv2.waitKey(0)
# misc.imshow(noisy_image)
# Destroy window and return to caller.
cv2.destroyAllWindows()
