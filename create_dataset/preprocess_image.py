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


def get_vid_name(file):
    """Creates the new video name.
      Args:
        file: String name of the original image name.
      Returns:
        A string name defined by the Subject and Session
        numbers.
    """

    return file.split("_")[0] + "_" + file.split("_")[1]



def get_label(filepath):
    """Returns the label for a given Session by reading its
        corresponding CSV label file.
      Args:
        filepath: String path to the CSV file.
      Returns:
        String label name if found, else a -1.
    """

    if os.path.exists(filepath) and os.listdir(filepath):
        g = open(filepath + str(os.listdir(filepath)[0]), 'r')
        label = g.readline().split('.')[0].replace(" ", "")
        return label
    else:
        return -1


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    '''
     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4
    '''
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top_right
    val_ar.append(get_pixel(img, center, x, y + 1))  # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y))  # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom_left
    val_ar.append(get_pixel(img, center, x, y - 1))  # left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top_left
    val_ar.append(get_pixel(img, center, x - 1, y))  # top

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val


def lbp_face(img, dest, vid_name, fr_mode, emotion, augment_mode, angle=None, norm=False):
    height, width, channel = img.shape
    #    print(img.shape)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)


    if norm == False:
        dest = dest + '/lbp/'
        if not os.path.exists(dest):
            os.makedirs(dest)
    if norm == True:
        dest = dest + '/norm_lbp/'
        if not os.path.exists(dest):
            os.makedirs(dest)

    img_lbp = cv2.resize(img_lbp, (128, 128), cv2.INTER_LINEAR)  # resize
    cv2.imwrite(dest + '/' + vid_name + '.png', img_lbp)

    return img_lbp # resized


def normalization(image, dest, vid_name, fr_mode, emotion, augment_mode, angle=None):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg = image.mean(axis=0).mean(axis=0)

    R, G, B = cv2.split(image)
    std = (R.std(), G.std(), B.std())

    new_image = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # print(image.std())
            new_image[i][j] = (image[i][j] - avg)/std

    # normalize float into int8(0~255)
    int_image = cv2.normalize(src=new_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    image = int_image
    image = cv2.resize(image, (128, 128), cv2.INTER_LINEAR) # resize

    # save resized image
    dest = dest + '/normalized/'
    if not os.path.exists(dest):
        os.makedirs(dest)
    cv2.imwrite(dest + '/' + vid_name + '.png', image)

    return int_image # not resized


def crop_face(image_path):
    """Returns a facial image if face is detected.

        If the FaceBoxes finds a face in the image, the returned
        image size will be defined by the bounding box returned by the face
        detector. Otherwise, the imagewill be centered cropped by empirically
        found parameters.
      Args:
        image_path: String path to the image file.
      Returns:
        OpenCV image object of the processed image.
    """

    print("crop image path : ", image_path)

    caffe_root = '/home/sjpark/FER/create_dataset'
    os.chdir(caffe_root)
    sys.path.insert(0, 'python')
    import caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()
    model_def = '/home/sjpark/FER/create_dataset/FaceBoxes/models/faceboxes/deploy.prototxt'
    model_weights = '/home/sjpark/FER/create_dataset/FaceBoxes/models/faceboxes/faceboxes.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    image = caffe.io.load_image(image_path)
    im_scale = 1.0
    if im_scale != 1.0:
        image = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    net.blobs['data'].reshape(1, 3, image.shape[0], image.shape[1])
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    detections = net.forward()['detection_out']
    detected = detections[0][0]
    sort_detected = sorted(detected, key=itemgetter(2), reverse=True)
    # for d in range(detected.shape[0]):
    for d in range(1):
        if sort_detected[d][1] < 0:
            print("Face could not be detected in image",
                  image_path + ", " + "proceeding with center cropping.")
            xmax = image.shape[0]
            xmin = 0
            ymax = image.shape[1]
            ymin = 0

            width = xmax - xmin
            height = ymax - ymin
        else:

            det_label = sort_detected[d][1]
            det_conf = sort_detected[d][2]
            det_xmin = sort_detected[d][3]
            det_ymin = sort_detected[d][4]
            det_xmax = sort_detected[d][5]
            det_ymax = sort_detected[d][6]

            if det_conf < 0:
                print("Face could not be detected in image",
                      image_path + ", " + "proceeding with center cropping.")
                xmax = image.shape[0]
                xmin = 0
                ymax = image.shape[1]
                ymin = 0

                width = xmax - xmin
                height = ymax - ymin
            else:
                # for i in range(top_conf.shape[0]):
                for i in range(1):
                    xmin = int(round(det_xmin * image.shape[1]))
                    ymin = int(round(det_ymin * image.shape[0]))
                    xmax = int(round(det_xmax * image.shape[1]))
                    ymax = int(round(det_ymax * image.shape[0]))

                    score = det_conf
                    display_txt = '%.2f' % (score)
                    # display_wh = '%.2f %.2f' % (xmax - xmin + 1, ymax - ymin + 1)


                    image = cv2.imread(image_path)
                    h, w, _ = image.shape
                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    if xmax > w:
                        xmax = w
                    if ymax > h:
                        ymax = h

                    width = xmax - xmin
                    height = ymax - ymin

    return [xmin, ymin, xmax, ymax]


def extract_geometry(path, prediction, neutral_image_path=True):

    emo_vid = io.imread(path)
    emo_img1 = emo_vid[:,:]

    if (len(emo_img1) is not 0) :
        # face detection by FaceBoxes
        rects = crop_face(path)

        # face alignment
        # make nose point and mouth point perpendicular
        fa = FaceAligner(prediction, desiredFaceWidth=abs(rects[0]-rects[2]), desiredFaceHeight=abs(rects[1]-rects[3]))
        faceAligned, leftEyePts, rightEyePts, nosePts = fa.align(emo_vid)

        # check whether the image is frontal or not
        if nosePts[0] >= leftEyePts[0][2] and nosePts[0] <= rightEyePts[0][0]:
            faceAligned = faceAligned
        else:
            faceAligned = emo_img1

        return faceAligned

    else:
        print('Cannot Capture Face')
        return -1


def align_face(image_path, dest, vid_name, fr_mode, emotion, augment_mode, angle=None):
    """Returns a facial image if face is detected.

        If the MTCNN face detector finds a face in the image, the returned
        image size will be defined by the bounding box returned by the face
        detector. Otherwise, the imagewill be centered cropped by empirically
        found parameters.
      Args:
        image_path: String path to the image file.
        detector: MTCNN object
      Returns:
        OpenCV image object of the processed image.
    """
    image = cv2.imread(image_path)

    saved_dir = '{}/'.format(dest) # '../CKP/Images/3/S005/' #'augment_emotion/slow/'
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    # landmark detection
    model = '/home/sjpark/FER/create_dataset/Alignment/SAN/snapshots/SAN_AFLW_FULL_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar'
    prediction = evaluate(image_path, model, False)

    # detection fail
    if (type(prediction) == int):
        pass

    else:
        # face detection + crop + alignment
        alignedImage = extract_geometry(image_path, prediction, model)
        alignedImage = cv2.cvtColor(alignedImage, cv2.COLOR_BGR2RGB)
        image = alignedImage
        image = cv2.resize(image, (128, 128), cv2.INTER_LINEAR) # resize


    dest = dest + '/preprocessed'
    if not os.path.exists(dest):
        os.makedirs(dest)
    cv2.imwrite(dest + '/' + vid_name + '.png', image)

    return alignedImage # not resized image


def preprocess_images(images_list, src, dest, subject, min_seq_len, t_height, t_width):
    """Creates the Numpy binary files with a 3D array of a sequence of images.
      Args:
        images_list: String array with paths to all images from a given Session.
        src: String path to a selected images Session.
        dest: String path to where the emotion-labeled Numpy binary files must
            be saved.
        neu_dest: String path to where the neutral-labeled Numpy binary files
        must be saved.
        min_seq_len: Integer number of minimum images in a given Session for them
            to be processed. Only Sessions with more than this value will be
            considered.
        depth: Integer number of images to be appended to create a 3D array.
            This is the temporal depth of the input instances.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
        crop_faces_flag: Boolean value that indicates if facial cropping will be
            done before resizing the original images.
        detector: MTCNN object.
    """

    # all images
    augment_mode = 0
    fr_mode = 0
    emo = 0

    vid_name = os.path.basename(images_list[0])

    print("vid name : ", vid_name)

    for index, image in enumerate(images_list):

        if not os.path.exists(dest + '/{}/norm_lbp/'.format(subject) + image + '.png'):
            im_size = cv2.imread(src + images_list[0]).shape
            im_shape = im_size
            vid = np.zeros(0)

            # src = '../CKP/extended-cohn-kanade-images/cohn-kanade-images/S005/001/'
            # image = S005_001_00000001.png
            im = align_face(src + image, dest + '/' + subject, image, fr_mode, emo, augment_mode)
            im_norm = normalization(im, dest + '/' + subject, image, fr_mode, emo, augment_mode)
            im_lbp = lbp_face(im, dest + '/' + subject, image, fr_mode, emo, augment_mode)
            im_normlbp = lbp_face(im_norm, dest + '/' + subject, image, fr_mode, emo, augment_mode, norm=True)

                


def gen_emotionsFolders(label_main_dir, image_main_dir, emotions_dir,
                        min_seq_len, t_height, t_width):
    """Generates a folder of Numpy binary files with 3D arrays of images for each
        label in CK+ database.
        Check each Session if the number of images is more than min_seq_len and
        create a 3D array by appending a depth number of images together. Each
        image is resized to (t_height, t_width) from its original size or from
        a face bounding box if the flag crop_faces is true. Numpy binary files
        are saved in emotions_dir.
      Args:
        label_main_dir: String path to CK+ labels folder
        image_main_dir: String path to CK+ images folder
        emotions_dir: String path where to Numpy binary files will be saved.
        crop_faces_flag: Boolean value that indicates if facial cropping will be
            done before resizing the original images.
        neutral_label: String name of the neutral label. This value has to be
            defined by the user because it will not appear in the CSV files.
        min_seq_len: Integer number of minimum images in a given Session for them
            to be processed. Only Sessions with more than this value will be
            considered.
        depth: Integer number of images to be appended to create a 3D array.
            This is the temporal depth of the input instances.
        t_height: Integer number of the target height to which the images will
            be resized.
        t_width: Integer number of the target width to which the images will
            be resized.
    """

    start_time = time.time()
    print("Getting images...")

    list_labels = np.array([])
    im_shape = []
    
    for subject in sorted(os.listdir(image_main_dir), key=last_3chars):
        for session in sorted(
                os.listdir(image_main_dir + str(subject)), key=last_3chars):
            if session != ".DS_Store":
                images_path = image_main_dir + str(subject) + '/' + str(
                    session) + '/'
                images_list = [
                    x
                    for x in sorted(os.listdir(images_path), key=last_8chars)
                    if x.split(".")[1] == "png"
                ]
                if (images_list and len(images_list) >= min_seq_len):
                    label_path = label_main_dir + str(subject) + '/' + str(
                        session) + '/'
                    label = get_label(label_path)
                    print("label ------------------------------ ", label)
                    if label != -1:
                        if label not in list_labels:
                            list_labels = np.append(list_labels, label)
                            if not os.path.exists(emotions_dir + str(label)):
                                os.makedirs(emotions_dir + str(label))

                        # original image
                        im_shape = preprocess_images(
                            images_list, images_path,
                            emotions_dir + str(label), subject, min_seq_len,
                            t_height, t_width)

    duration = time.time() - start_time
    print("\nDone! Total time %.1f seconds." % (duration))


def main(argv):
    gen_emotionsFolders(
        label_main_dir=str(argv[0]),  # ckp/emotion_labels
        image_main_dir=str(argv[1]),  # ckp/cohn-kanade-images/ : 변?�할 ?��?지 ?�더
        emotions_dir=str(argv[2]),  # ckp/emotion_images/ : ?�로 만들?�질 ?�더
        neutral_label=str(argv[3]),  # 0
        min_seq_len=int(argv[4]),  # 9
        t_height=int(argv[5]),  # 128
        t_width=int(argv[6]))  # 128


if __name__ == "__main__":
    main(sys.argv[1:])