# -*- coding: utf-8 -*-

import os, sys, random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import shutil
from operator import itemgetter
from landmark_detection.SAN.san_eval import evaluate
from skimage import io as io
from imutils.face_utils import FaceAligner


def last_15chars(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 15 characters of x.
    """

    return (x[-15:])


def last_8chars(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 8 characters of x.
    """

    return (x[-8:])


def last_3chars(x):
    """Function that aids at list sorting.
      Args:
        x: String name of files.
      Returns:
        A string of the last 8 characters of x.
    """

    return (x[-3:])


def create_moving_batch(images_list, slow_frame, fast_frame, middle_frame):
    """Creates the list of neutral and emotion-labeled images to be processed.
        From the whole list of images of a given session, create an 2 dimensional
        array with neutral and emotion-labeled image names. The quantity of
        images per index in batch (same for neutral and emotion) is defined by
        the selected clip depth. 
      Args:
        images_list: String array with paths to all images from a given session.
        slow_frame: The number of choosing slow-labed images in sequence
        fast_frame: The number of choosing fast-labed images in sequence
        middle_frame: The number of choosing middle-labed images in sequence
      Returns:
        A 2D string array containing the path to the images to be processed later
        to create 3D arrays:
            batch1 = slow-labeled images
            batch2 = fast-labeled images
            batch2 = middle-labeled images
            batch1[0] = slow-emotion-labeled images
            batch1[1] = slow-neutral-labeled images
            batch2[0] = fast-emotion-labeled images
            batch2[1] = fast-neutral-labeled images
            batch3[0] = middle-emotion-labeled images
            batch3[1] = middle-neutral-labeled images
    """

    print("image_list size : ", len(images_list))

    batch1 = np.zeros((2, slow_frame), dtype='<U32')
    batch2 = np.zeros((2, fast_frame), dtype='<U32')
    batch3 = np.zeros((2, middle_frame), dtype='<U32')

    # define the intervals
    sinterval = (len(images_list) - 1) / (slow_frame - 1)
    finterval = (len(images_list) - 1) / (fast_frame - 1)
    minterval = (len(images_list) - 1) / (middle_frame - 1)

    for i in range(slow_frame):
        for index, image in enumerate(sorted(images_list, key=last_8chars)):
            # neutral images
            if (index < slow_frame):
                batch1[1][index] = image
            
            # other emotion images
            if index == 0:
                batch1[0][0] = image
                i += 1
            elif index == round(sinterval * i):
                batch1[0][i] = image
                i += 1

    for i in range(fast_frame):
        for index, image in enumerate(sorted(images_list, key=last_8chars)):
            # nuetral images
            if (index == 0):
                batch2[1][0] = image
                batch2[1][1] = image
            elif (index == 1):
                batch2[1][2] = image
                batch2[1][3] = image
            elif (index == 2):
                batch2[1][4] = image
                batch2[1][5] = image
                batch2[1][6] = image
            
            # other emotion images
            if index == 0:
                batch2[0][0] = image
                i += 1
            elif index == round(finterval * i):
                batch2[0][i] = image
                i += 1

    for i in range(middle_frame):
        for index, image in enumerate(sorted(images_list, key=last_8chars)):
            # nuetral images
            if (index == 0):
                batch3[1][0] = image
            elif (index == 1):
                batch3[1][1] = image
                batch3[1][2] = image
            elif (index == 2):
                batch3[1][3] = image
                batch3[1][4] = image

            # other emotion images
            if index == 0:
                batch3[0][0] = image
                i += 1
            elif index == round(minterval * i):
                batch3[0][i] = image
                i += 1

    return batch1, batch2, batch3


def move_images(images_list, src, dest, t_height, t_width):
    """Creates the Numpy binary files with a 3D array of a sequence of images.
      Args:
        images_list: String array with paths to all images from a given Session.
        src: String path to a selected images Session.
        dest: String path to where the emotion-labeled Numpy binary files must
            be saved.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
    """

    slow_frame = 3
    fast_frame = 7
    middle_frame = 5

    # Create neutral and emotion-labeled lists to be processed.
    batch1, batch2, batch3 = create_moving_batch(images_list, slow_frame, fast_frame, middle_frame)

    # Slow batch
    for i in range(batch1.shape[0]):

        vid_name = os.path.basename(batch1[i][0])
        print("vid name : ", vid_name)

        if not os.path.exists(dest + '/n-s-' + vid_name + '.npy'):

            im_size = cv2.imread(src + batch1[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch1[i]):
                # print(index, " : " , image)

                # src = '../CKP/extended-cohn-kanade-images/cohn-kanade-images/S005/001/'
                # image = S005_001_00000001.png
                im = cv2.imread(src + image)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

                if i == 0:
                    npyname = 's-' + batch1[i][0]
                if i == 1:
                    npyname = 'n-s-' + batch1[i][0]

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]))
                raise
            # Save to Numpy binary file'''

    # Fast batch
    for i in range(batch2.shape[0]):

        vid_name = os.path.basename(batch2[i][0])

        print("vid name : ", vid_name)

        if not os.path.exists(dest + '/n-f-' + vid_name + '.npy'):

            im_size = cv2.imread(src + batch2[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch2[i]):
                # print(index, " : " , image)

                # src = '../CKP/extended-cohn-kanade-images/cohn-kanade-images/S005/001/'
                # image = S005_001_00000001.png
                im = cv2.imread(src + image)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

                if i == 0:
                    npyname = 'f-' + batch2[i][0]
                if i == 1:
                    npyname = 'n-f-' + batch2[i][0]

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]))
                raise
            # Save to Numpy binary file'''

    # Middle batch
    for i in range(batch3.shape[0]):

        vid_name = os.path.basename(batch3[i][0])

        print("vid name : ", vid_name)

        if not os.path.exists(dest + '/n-m-' + vid_name + '.npy'):

            im_size = cv2.imread(src + batch3[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch3[i]):
                # print(index, " : " , image)

                # src = '../CKP/extended-cohn-kanade-images/cohn-kanade-images/S005/001/'
                # image = S005_001_00000001.png
                im = cv2.imread(src + image)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

                if i == 0:
                    npyname = 'm-' + batch3[i][0]
                if i == 1:
                    npyname = 'n-m-' + batch3[i][0]

            try:

                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]))
                raise
            # Save to Numpy binary file'''


def move_images_with_flip(images_list, src, dest, t_height, t_width):
    """Creates the Numpy binary files with a 3D array of a sequence of horizontally flipped images.
      Args:
        images_list: String array with paths to all images from a given Session.
        src: String path to a selected images Session.
        dest: String path to where the emotion-labeled Numpy binary files must
            be saved.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
    """
    

    slow_frame = 3
    fast_frame = 7
    middle_frame = 5

    # Create neutral and emotion-labeled lists to be processed.
    batch1, batch2, batch3 = create_moving_batch(images_list, slow_frame, fast_frame, middle_frame)

    from numpy import expand_dims

    # Slow batch
    for i in range(batch1.shape[0]):

        vid_name = os.path.basename(batch1[i][0])

        if not os.path.exists(dest + '/s-' + vid_name + '_flip.npy'):
            im_size = cv2.imread(src + batch1[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch1[i]):
                im = cv2.imread(src + image)
                im = np.fliplr(im)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                npyname = 's-' + batch1[i][0] + '_flip'
            if i == 1:
                npyname = 'n-s-' + batch1[i][0] + '_flip'

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_flip")
                raise

    # Fast batch
    for i in range(batch2.shape[0]):

        vid_name = os.path.basename(batch2[i][0])

        if not os.path.exists(dest + '/f-' + vid_name + '_flip.npy'):
            im_size = cv2.imread(src + batch2[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch2[i]):
                im = cv2.imread(src + image)
                im = np.fliplr(im)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                npyname = 'f-' + batch2[i][0] + '_flip'
            if i == 1:
                npyname = 'n-f-' + batch2[i][0] + '_flip'

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_flip")
                raise

    # Middle batch
    for i in range(batch3.shape[0]):

        vid_name = os.path.basename(batch3[i][0])

        if not os.path.exists(dest + '/m-' + vid_name + '_flip.npy'):
            im_size = cv2.imread(src + batch3[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch3[i]):
                im = cv2.imread(src + image)
                im = np.fliplr(im)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                npyname = 'm-' + batch3[i][0] + '_flip'
            if i == 1:
                npyname = 'n-m-' + batch3[i][0] + '_flip'

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_flip")
                raise


def rotation(im, angle):
    resizedImg = cv2.resize(im, (int(im.shape[1] * 0.9), int(im.shape[0] * 0.9)))

    scaleFactor = 1
    degreesCCW = angle
    oldY = resizedImg.shape[0]
    oldX = resizedImg.shape[1]  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv2.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=angle,
                                scale=scaleFactor)  # rotate about center of image.

    # choose a new image size.
    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    # include this if you want to prevent corners being cut off
    r = np.deg2rad(degreesCCW)
    newX, newY = (
        abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))

    # So I will find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv2.warpAffine(resizedImg, M, dsize=(int(newX), int(newY)))

    return rotatedImg


def move_images_with_rotate(images_list, src, dest, t_height, t_width, angle):
    """Creates the Numpy binary files with a 3D array of a sequence of rotated images.
      Args:
        images_list: String array with paths to all images from a given Session.
        src: String path to a selected images Session.
        dest: String path to where the emotion-labeled Numpy binary files must
            be saved.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
        angle: A number indicating how much to rotate
    """
    
    # Create neutral and emotion-labeled lists to be processed.
    slow_frame = 3
    fast_frame = 7
    middle_frame = 5

    # Create neutral and emotion-labeled lists to be processed.
    batch1, batch2, batch3 = create_moving_batch(images_list, slow_frame, fast_frame, middle_frame)

    from numpy import expand_dims

    # Slow batch
    for i in range(batch1.shape[0]):

        vid_name = os.path.basename(batch1[i][0])

        if not os.path.exists(dest + '/s-' + vid_name + 'rotation_75.npy'):
            im_size = cv2.imread(src + batch1[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch1[i]):
                im = cv2.imread(src + image)
                im = rotation(im, angle)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                if angle == -7.5:
                    npyname = 's-' + vid_name + 'rotation_-75'
                elif angle == -5:
                    npyname = 's-' + vid_name + 'rotation_-5'
                elif angle == -2.5:
                    npyname = 's-' + vid_name + 'rotation_-25'
                elif angle == 7.5:
                    npyname = 's-' + vid_name + 'rotation_75'
                elif angle == 5:
                    npyname = 's-' + vid_name + 'rotation_5'
                elif angle == 2.5:
                    npyname = 's-' + vid_name + 'rotation_25'
            if i == 1:
                pass

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_rotation")
                raise

    # Fast batch
    for i in range(batch2.shape[0]):

        vid_name = os.path.basename(batch2[i][0])

        if not os.path.exists(dest + '/f-' + vid_name + 'rotation_75.npy'):
            im_size = cv2.imread(src + batch2[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch2[i]):
                im = cv2.imread(src + image)
                im = rotation(im, angle)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                if angle == -7.5:
                    npyname = 'f-' + vid_name + 'rotation_-75'
                elif angle == -5:
                    npyname = 'f-' + vid_name + 'rotation_-5'
                elif angle == -2.5:
                    npyname = 'f-' + vid_name + 'rotation_-25'
                elif angle == 7.5:
                    npyname = 'f-' + vid_name + 'rotation_75'
                elif angle == 5:
                    npyname = 'f-' + vid_name + 'rotation_5'
                elif angle == 2.5:
                    npyname = 'f-' + vid_name + 'rotation_25'
            if i == 1:
                pass

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_rotation")
                raise

    # Middle batch
    for i in range(batch3.shape[0]):

        vid_name = os.path.basename(batch3[i][0])

        if not os.path.exists(dest + '/m-' + vid_name + 'rotation_75.npy'):
            im_size = cv2.imread(src + batch3[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch3[i]):
                im = cv2.imread(src + image)
                im = rotation(im, angle)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                if angle == -7.5:
                    npyname = 'm-' + vid_name + 'rotation_-75'
                elif angle == -5:
                    npyname = 'm-' + vid_name + 'rotation_-5'
                elif angle == -2.5:
                    npyname = 'm-' + vid_name + 'rotation_-25'
                elif angle == 7.5:
                    npyname = 'm-' + vid_name + 'rotation_75'
                elif angle == 5:
                    npyname = 'm-' + vid_name + 'rotation_5'
                elif angle == 2.5:
                    npyname = 'm-' + vid_name + 'rotation_25'
            if i == 1:
                pass

            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_rotation")
                raise



def move_images_with_flip_rotate(images_list, src, dest, t_height, t_width, angle):
    """Creates the Numpy binary files with a 3D array of a sequence of horizontally flipped and rotated images.
      Args:
        images_list: String array with paths to all images from a given Session.
        src: String path to a selected images Session.
        dest: String path to where the emotion-labeled Numpy binary files must
            be saved.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
        angle: A number indicating how much to rotate
    """
    
    # Create neutral and emotion-labeled lists to be processed.
    slow_frame = 3
    fast_frame = 7
    middle_frame = 5

    # Create neutral and emotion-labeled lists to be processed.
    batch1, batch2, batch3 = create_moving_batch(images_list, slow_frame, fast_frame, middle_frame)

    from numpy import expand_dims

    # Slow batch
    for i in range(batch1.shape[0]):

        vid_name = os.path.basename(batch1[i][0])

        if not os.path.exists(dest + '/s-' + vid_name + 'flip_rotation_75.npy'):
            im_size = cv2.imread(src + batch1[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch1[i]):
                im = cv2.imread(src + image)
                im = np.fliplr(im)
                im = rotation(im, angle)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                if angle == -7.5:
                    npyname = 's-' + vid_name + 'flip_rotation_-75'
                elif angle == -5:
                    npyname = 's-' + vid_name + 'flip_rotation_-5'
                elif angle == -2.5:
                    npyname = 's-' + vid_name + 'flip_rotation_-25'
                elif angle == 7.5:
                    npyname = 's-' + vid_name + 'flip_rotation_75'
                elif angle == 5:
                    npyname = 's-' + vid_name + 'flip_rotation_5'
                elif angle == 2.5:
                    npyname = 's-' + vid_name + 'flip_rotation_25'
            if i == 1:
                pass
                
            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_flip_rotation")
                raise

    # Fast batch
    for i in range(batch2.shape[0]):

        vid_name = os.path.basename(batch2[i][0])

        if not os.path.exists(dest + '/f-' + vid_name + 'flip_rotation_75.npy'):
            im_size = cv2.imread(src + batch2[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch2[i]):
                im = cv2.imread(src + image)
                im = np.fliplr(im)
                im = rotation(im, angle)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                if angle == -7.5:
                    npyname = 'f-' + vid_name + 'flip_rotation_-75'
                elif angle == -5:
                    npyname = 'f-' + vid_name + 'flip_rotation_-5'
                elif angle == -2.5:
                    npyname = 'f-' + vid_name + 'flip_rotation_-25'
                elif angle == 7.5:
                    npyname = 'f-' + vid_name + 'flip_rotation_75'
                elif angle == 5:
                    npyname = 'f-' + vid_name + 'flip_rotation_5'
                elif angle == 2.5:
                    npyname = 'f-' + vid_name + 'flip_rotation_25'
            if i == 1:
                pass
                
            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "_flip_rotation")
                raise

    # Middle batch
    for i in range(batch3.shape[0]):

        vid_name = os.path.basename(batch3[i][0])

        if not os.path.exists(dest + '/m-' + vid_name + 'flip_rotation_75.npy'):
            im_size = cv2.imread(src + batch3[i][0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            for index, image in enumerate(batch3[i]):
                im = cv2.imread(src + image)
                im = np.fliplr(im)
                im = rotation(im, angle)
                im_resized = cv2.resize(im, (t_height, t_width), cv2.INTER_LINEAR)

                if index == 0:
                    temp = np.zeros((0, im_resized.shape[0], im_resized.shape[1],
                                     im_resized.shape[2]))

                temp = np.append(temp, [im_resized], axis=0)
                vid = temp

            # Save to Numpy binary file
            if i == 0:
                if angle == -7.5:
                    npyname = 'm-' + vid_name + 'flip_rotation_-75'
                elif angle == -5:
                    npyname = 'm-' + vid_name + 'flip_rotation_-5'
                elif angle == -2.5:
                    npyname = 'm-' + vid_name + 'flip_rotation_-25'
                elif angle == 7.5:
                    npyname = 'm-' + vid_name + 'flip_rotation_75'
                elif angle == 5:
                    npyname = 'm-' + vid_name + 'flip_rotation_5'
                elif angle == 2.5:
                    npyname = 'm-' + vid_name + 'flip_rotation_25'
            if i == 1:
                pass
                
            try:
                np.save(dest + npyname, vid)
                print(dest + npyname, " is saved")

            except Exception as e:
                print("Unable to save instance at:",
                      dest + str(vid_name[0]) + "_" + str(vid_name[1]) + "flip_rotation")
                raise



def gen_emotionsFolders(image_main_dir, emotions_dir,
                        neutral_label, t_height, t_width):
    """Generates a folder of Numpy binary files with 3D arrays of images for each
        label in CK+ database.
        Create a 3D array by appending a depth number of images together. Each
        image is resized to (t_height, t_width) from its original size or from
        a face bounding box if the flag crop_faces is true. Numpy binary files
        are saved in emotions_dir.
      Args:
        image_main_dir: String path to CK+ images folder
        emotions_dir: String path where to Numpy binary files will be saved.
        neutral_label: String name of the neutral label. This value has to be
            defined by the user because it will not appear in the CSV files.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
    """

    start_time = time.time()
    print("Getting images...")

    list_labels = np.array([])
    im_shape = []
    list_labels = np.append(list_labels, neutral_label)

    for subject in sorted(os.listdir(image_main_dir), key=last_3chars): # subject:'1'
        for session in sorted( # session:'S010'
                os.listdir(image_main_dir + str(subject)), key=last_3chars):
            if session != ".DS_Store":
                folder_path = image_main_dir + str(subject) + '/' + str(session) + '/' # folder_path : '../CKP/PreprocessedImages/1/S101/'
                folder_list = [
                    x
                    for x in sorted(os.listdir(folder_path), key=last_8chars)
                ]
                for index, folder in enumerate(folder_list):
                    dest = emotions_dir

                    if folder == 'lbp':
                        if not os.path.exists(dest + 'lbp/' + neutral_label + '/'):
                            os.makedirs(dest + 'lbp/' + neutral_label + '/')
                        dest = dest + 'lbp/' + subject + '/'
                    if folder == 'nlbp':
                        if not os.path.exists(dest + 'nlbp/' + neutral_label + '/'):
                            os.makedirs(dest + 'nlbp/' + neutral_label + '/')
                        dest = dest + 'nlbp/' + subject + '/'
                    if folder == 'norm_lbp':
                        if not os.path.exists(dest + 'norm_lbp/' + neutral_label + '/'):
                            os.makedirs(dest + 'norm_lbp/' + neutral_label + '/')
                        dest = dest + 'norm_lbp/' + subject + '/'
                    if folder == 'norm_nlbp':
                        if not os.path.exists(dest + 'norm_nlbp/' + neutral_label + '/'):
                            os.makedirs(dest + 'norm_nlbp/' + neutral_label + '/')
                        dest = dest + 'norm_nlbp/' + subject + '/'
                    if folder == 'normalized':
                        if not os.path.exists(dest + 'normalized/' + neutral_label + '/'):
                            os.makedirs(dest + 'normalized/' + neutral_label + '/')
                        dest = dest + 'normalized/' + subject + '/'
                    if folder == 'preprocessed':
                        if not os.path.exists(dest + 'preprocessed/' + neutral_label + '/'):
                            os.makedirs(dest + 'preprocessed/' + neutral_label + '/')
                        dest = dest + 'preprocessed/' + subject + '/'

                    if not os.path.exists(dest):
                        os.makedirs(dest)

                    images_path = image_main_dir + str(subject) + '/' + str(session) + '/' + str(folder) + '/'
                    images_list = [
                        x
                        for x in sorted(os.listdir(images_path), key=last_15chars)
                        if x.split(".")[-1] == "png"
                    ]

                    label = subject
                    print("label ------------------------------ ", label)

                    if label != -1:
                        if label not in list_labels:
                            list_labels = np.append(list_labels, label)


                        # original image
                        im_shape = move_images(
                            images_list, images_path, dest, t_height, t_width)

                        # flipped image
                        im_shape = move_images_with_flip(
                            images_list, images_path, dest, t_height, t_width)

                        angles = [-2.5, -5, -7.5, 2.5, 5, 7.5]
                        # rotated image
                        for angle in angles:
                            im_shape = move_images_with_rotate(
                                images_list, images_path, dest, t_height, t_width, angle)
                                
                        # flipped&rotated image
                        for angle in angles:
                            im_shape = move_images_with_flip_rotate(
                                images_list, images_path, dest, t_height, t_width, angle)


    duration = time.time() - start_time
    print("\nDone! Total time %.1f seconds." % (duration))
    print("dir :", emotions_dir + neutral_label)
    print(type(emotions_dir), emotions_dir)
    print(type(neutral_label), neutral_label)
    print("list : ", os.listdir(emotions_dir + neutral_label))
    print("######################", sorted(os.listdir(emotions_dir + neutral_label))[0])
    test_vid = np.load(emotions_dir + neutral_label + "/" + sorted(os.listdir(emotions_dir + neutral_label))[0])
    print("Clip size:", test_vid.shape)
    print("Neutral label example:")
    plt.imshow(test_vid[0][:, :, 0], cmap="gray", interpolation="nearest")
    plt.axis("off")
    plt.show()


def main(argv):
    gen_emotionsFolders(
        image_main_dir=str(argv[0]),  # ckp/cohn-kanade-images/ : è¹‚Â€?? ì?ë¸??? ì™??ï§žÂ€ ?? ìŽˆ??        
        emotions_dir=str(argv[1]),  # ckp/emotion_images/ : ?? ìŽˆì¤?ï§ëš®ë±?? ìŽŒì­??? ìŽˆ??        
        neutral_label=str(argv[2]),  # 0
        t_height=int(argv[3]),  # 128
        t_width=int(argv[4]))  # 128


if __name__ == "__main__":
    main(sys.argv[1:])