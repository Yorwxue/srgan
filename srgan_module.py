#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl

from models.srgan import *
from utils import *
import config
from config import config, log_config
from scipy import misc
import cv2

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def read_all_imgs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=path)
        # print(b_imgs.shape)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs


def sharp(img_path):
    print('STRAT')
    ## create folders to save result images
    save_dir = "samples/evaluate"
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    # ###========================== DEFINE MODEL ============================###
    img = misc.imread(img_path)
    img = img[:, :, 0:3]

    valid_lr_img = img
    # valid_lr_img = cv2.resize(valid_lr_img, (0, 0), fx=0.25, fy=0.25)

    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    size = valid_lr_img.shape
    t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image')
    #
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    # ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (
    size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    # tl.vis.save_image(img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')

if __name__ == '__main__':
    img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/photo_image/small/2231252.png'
    # img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/face_image/test_lr/11.jpg'
    # img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/face_image/pic_of_person_lr/涂滿煌.png'
    # img_path = '/media/clliao/9c88dfb2-c12d-48cc-b30b-eaffb0cbf545/SRGAN_dataset/face_image/valid_lr/Aaron_Sorkin_0002.png'

    sharp(img_path)
