'''
from PIL import Image

image = Image.open('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-Keras-master/AOI_Trace/Review/PAD_PASS/PAD_44919_186871.png')
resize_image = image[1600:1600]

resize_image.save('./test.jpg')

'''

import cv2
import numpy as np
import os




# crop & resize
for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/Train_Val_ksh/review/Train/ALL'):
    for fname in files:
        original_img = cv2.imread(root+'/'+fname)
        crop_img = original_img[0:1600, :, :]
        crop_img = np.array(crop_img, 'uint8')

        '''
        #HR Gray Version
        HR_train_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        HR_train_img = cv2.resize(HR_train_img, dsize=(480, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/HJ_trainSet/ver4_review_AB_train/matched_review_480_gray_HR' + '/' + fname, HR_train_img)
        '''
        # HR Gray Version
        #HR Color Version


        '''
        HR_train_img = cv2.resize(crop_img, dsize=(120, 120), interpolation=cv2.INTER_AREA)

        cv2.imwrite('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/temp'+'/'+fname, HR_train_img)
        '''


        '''
        #HR Color Version
        #HR_train_img = cv2.resize(crop_img, dsize=(480, 480), interpolation=cv2.INTER_AREA)
        cv2.imwrite('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/Review_480_gray_HR'+'/'+fname, HR_train_img)
        '''


        LR_train_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        LR_train_img = cv2.resize(LR_train_img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        #cv2.imwrite('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/test'+'/'+fname, LR_train_img)
        cv2.imwrite(
            '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/' + '256' + fname,
            LR_train_img)




'''
# crop & 1 channel gray image & resize
for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-Keras-master/AOI_Trace/Review_1cGray'):
    category = root.split('/')[-1]
    for fname in files:

        original_img = cv2.imread(root + '/' + fname)
        new_img = original_img[0:1600, :, :]
        new_img = np.array(new_img, 'uint8')
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, dsize=(120, 120), interpolation=cv2.INTER_AREA)
        cv2.imwrite(root+'/'+fname, gray)

'''

'''
# crop & 3 channel gray image
for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-Keras-master/AOI_Trace/Review_3cGray'):
    category = root.split('/')[-1]
    for fname in files:

        original_img = cv2.imread(root + '/' + fname)
        new_img = original_img[0:1600, :, :]
        new_img = np.array(new_img, 'uint8')
        gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        #gray = cv2.resize(gray, dsize=(120, 120), interpolation=cv2.INTER_AREA)
        final_shape=(120,120,3)
        img2 = np.zeros_like(new_img)
        img2[:, :, 0] = gray
        img2[:, :, 1] = gray
        img2[:, :, 2] = gray
        gray = cv2.resize(gray, dsize=(120, 120), interpolation=cv2.INTER_AREA)
        cv2.imwrite(root+'/'+fname, img2)
'''





'''
# crop & rgbtogray
for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-Keras-master/AOI_Trace/Review_gray'):
    category = root.split('/')[-1]
    for fname in files:
        original_img = cv2.imread(root+'/'+fname)
        new_img = original_img[0:1600, :, :]
        new_img = np.array(new_img, 'uint8')
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(root+'/'+fname, new_img)

'''
