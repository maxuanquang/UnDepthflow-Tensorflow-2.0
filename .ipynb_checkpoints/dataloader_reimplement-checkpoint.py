import os
import time 
import random
import numpy as np
import tensorflow as tf
import cv2
import imutils


def string_length_tf(t):
    return tf.compat.v1.py_func(len, [t], [tf.int64])


def rescale_intrinsics(raw_cam_mat, opt, orig_height, orig_width):
    fx = raw_cam_mat[0, 0]
    fy = raw_cam_mat[1, 1]
    cx = raw_cam_mat[0, 2]
    cy = raw_cam_mat[1, 2]
    r1 = tf.stack(
        [fx * opt.img_width / orig_width, 0, cx * opt.img_width / orig_width])
    r2 = tf.stack([
        0, fy * opt.img_height / orig_height, cy * opt.img_height / orig_height
    ])
    r3 = tf.constant([0., 0., 1.])
    return tf.stack([r1, r2, r3])

def get_multi_scale_intrinsics(raw_cam_mat, num_scales):
    proj_cam2pix = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = raw_cam_mat[0, 0] / (2**s)
        fy = raw_cam_mat[1, 1] / (2**s)
        cx = raw_cam_mat[0, 2] / (2**s)
        cy = raw_cam_mat[1, 2] / (2**s)
        r1 = tf.stack([fx, 0, cx])
        r2 = tf.stack([0, fy, cy])
        r3 = tf.constant([0., 0., 1.])
        proj_cam2pix.append(tf.stack([r1, r2, r3]))
    proj_cam2pix = tf.stack(proj_cam2pix)
    proj_pix2cam = tf.linalg.inv(proj_cam2pix)
    proj_cam2pix.set_shape([num_scales, 3, 3])
    proj_pix2cam.set_shape([num_scales, 3, 3])
    return proj_cam2pix, proj_pix2cam


def read_image(image_path, get_shape=False):
    # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
    file_extension = image_path.split()[-1]
    file_cond = tf.equal(file_extension, 'jpg')

    image = tf.cond(
        file_cond, lambda: tf.io.image.decode_jpeg(tf.read_file(image_path)),
        lambda: tf.io.image.decode_png(tf.read_file(image_path)))
    orig_height = tf.cast(tf.shape(image)[0], "float32")
    orig_width = tf.cast(tf.shape(image)[1], "float32")

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(
        image, [opt.img_height, opt.img_width],
        tf.image.ResizeMethod.AREA)

    if get_shape:
        return image, orig_height, orig_width
    else:
        return image
    
def augment_image_pair(left_image, right_image):
    # randomly shift gamma
    random_gamma = tf.random.uniform([], 0.8, 1.2)
    left_image_aug = left_image**random_gamma
    right_image_aug = right_image**random_gamma

    # randomly shift brightness
    random_brightness = tf.random.uniform([], 0.5, 2.0)
    left_image_aug = left_image_aug * random_brightness
    right_image_aug = right_image_aug * random_brightness

    # randomly shift color
    random_colors = tf.random.uniform([3], 0.8, 1.2)
    white = tf.ones([tf.shape(input=left_image)[0], tf.shape(input=left_image)[1]])
    color_image = tf.stack(
        [white * random_colors[i] for i in range(3)], axis=2)
    left_image_aug *= color_image
    right_image_aug *= color_image

    # saturate
    left_image_aug = tf.clip_by_value(left_image_aug, 0, 1)
    right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

    return left_image_aug, right_image_aug

def augment_image_list(image_list):
    # randomly shift gamma
    random_gamma = tf.random.uniform([], 0.8, 1.2)
    random_gamma = tf.cast(random_gamma, tf.float32)
    image_list = [img**random_gamma for img in image_list]

    # randomly shift brightness
    random_brightness = tf.random.uniform([], 0.5, 2.0)
    random_brightness = tf.cast(random_brightness, tf.float32)
    image_list = [img * random_brightness for img in image_list]

    # randomly shift color
    random_colors = tf.random.uniform([3], 0.8, 1.2)
    white = tf.ones(
        [tf.shape(input=image_list[0])[0], tf.shape(input=image_list[0])[1]])
    color_image = tf.stack(
        [white * random_colors[i] for i in range(3)], axis=2)
    image_list = [img * color_image for img in image_list]

    # saturate
    image_list = [tf.clip_by_value(img, 0, 1) for img in image_list]

    return image_list

def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0., 0., 1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def data_augmentation(im, intrinsics, out_h, out_w):
    # Random scaling
    def random_scaling(im, intrinsics):
        batch_size, in_h, in_w, _ = im.get_shape().as_list()
        scaling = tf.random.uniform([2], 1, 1.15)
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
        im = tf.compat.v1.image.resize(im, [out_h, out_w], method=tf.image.ResizeMethod.AREA)
        fx = intrinsics[:, 0, 0] * x_scaling
        fy = intrinsics[:, 1, 1] * y_scaling
        cx = intrinsics[:, 0, 2] * x_scaling
        cy = intrinsics[:, 1, 2] * y_scaling
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(input=im))
        offset_y = tf.random.uniform(
            [1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = tf.random.uniform(
            [1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
        im = tf.image.crop_to_bounding_box(im, offset_y, offset_x, out_h,
                                           out_w)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
        cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    im, intrinsics = random_scaling(im, intrinsics)
    im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
    return im, intrinsics


def generator_train(opt):
    filenames_file = opt.train_file
    data_path = opt.data_dir

    with open(filenames_file, 'r') as f:
        input_queue = f.readlines()
    split_line = [line.split() for line in input_queue]
    
    for line in split_line:
        left_image_path = r"/".join([data_path, line[0]])
        right_image_path = r"/".join([data_path, line[1]])
        next_left_image_path = r"/".join([data_path, line[2]])
        next_right_image_path = r"/".join([data_path, line[3]])
        cam_intrinsic_path = r"/".join([data_path, line[4]])
        
        left_image_o = cv2.imread(left_image_path)
        left_image_o = cv2.resize(left_image_o, (opt.img_width, opt.img_height))
        orig_height, orig_width = left_image_o.shape[:2]
        right_image_o = cv2.imread(right_image_path)
        right_image_o = cv2.resize(right_image_o, (opt.img_width, opt.img_height))
        next_left_image_o = cv2.imread(next_left_image_path)
        next_left_image_o = cv2.resize(next_left_image_o, (opt.img_width, opt.img_height))
        next_right_image_o = cv2.imread(next_right_image_path)
        next_right_image_o = cv2.resize(next_right_image_o, (opt.img_width, opt.img_height))
        
        # randomly flip images
        do_flip = tf.random.uniform([], 0, 1)
        left_image = tf.cond(pred=do_flip > 0.5,
                             true_fn=lambda: tf.compat.v1.image.flip_left_right(right_image_o),
                             false_fn=lambda: left_image_o)
        right_image = tf.cond(pred=do_flip > 0.5,
                              true_fn=lambda: tf.compat.v1.image.flip_left_right(left_image_o),
                              false_fn=lambda: right_image_o)
        
        next_left_image = tf.cond(
            pred=do_flip > 0.5,
            true_fn=lambda: tf.compat.v1.image.flip_left_right(next_right_image_o),
            false_fn=lambda: next_left_image_o)
        next_right_image = tf.cond(
            pred=do_flip > 0.5, true_fn=lambda: tf.compat.v1.image.flip_left_right(next_left_image_o),
            false_fn=lambda: next_right_image_o)
        
        # random shuffle order of image
        do_flip_fb = tf.random.uniform([], 0, 1)
        left_image, right_image, next_left_image, next_right_image = tf.cond(
            pred=do_flip_fb > 0.5,
            true_fn=lambda: (next_left_image, next_right_image, left_image, right_image),
            false_fn=lambda: (left_image, right_image, next_left_image, next_right_image)
        )
        
        #randomly augment images
#         do_augment  = tf.random.uniform([], 0, 1)
#         image_list = [left_image, right_image, next_left_image, next_right_image]
#         left_image, right_image, next_left_image, next_right_image = tf.cond(do_augment > 0.5, 
#                                                                              lambda: augment_image_list(image_list), 
#                                                                              lambda: image_list)

        # calculate raw camera matrix
        raw_cam_contents = tf.io.read_file(cam_intrinsic_path)
        last_line = tf.compat.v1.string_split(
            [raw_cam_contents], delimiter="\n").values[-1]
        raw_cam_vec = tf.compat.v1.strings.to_number(
            tf.compat.v1.string_split([last_line]).values[1:])
        raw_cam_mat = tf.reshape(raw_cam_vec, [3, 4])
        raw_cam_mat = raw_cam_mat[0:3, 0:3]
        raw_cam_mat = rescale_intrinsics(raw_cam_mat, opt, orig_height,
                                         orig_width)

        # Scale and crop augmentation
#         im_batch = tf.concat([tf.expand_dims(left_image, 0), 
#                          tf.expand_dims(right_image, 0),
#                          tf.expand_dims(next_left_image, 0),
#                          tf.expand_dims(next_right_image, 0)], axis=3)
#         raw_cam_mat_batch = tf.expand_dims(raw_cam_mat, axis=0)
#         im_batch, raw_cam_mat_batch = data_augmentation(im_batch, raw_cam_mat_batch, opt.img_height, opt.img_width)
#         left_image, right_image, next_left_image, next_right_image = tf.split(im_batch[0,:,:,:], num_or_size_splits=4, axis=2)
#         raw_cam_mat = raw_cam_mat_batch[0,:,:]
        
        # calculate projection
        proj_cam2pix, proj_pix2cam = get_multi_scale_intrinsics(raw_cam_mat,
                                                                opt.num_scales)

        yield left_image, right_image, next_left_image, next_right_image, proj_cam2pix, proj_pix2cam