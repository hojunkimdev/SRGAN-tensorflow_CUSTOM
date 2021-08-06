import os
import shutil

#f = open('/home/hojun/PycharmProjects/SR/SRGAN-Keras/AOI_Trace/raw.txt',encoding="ISO-8859-1")

'''

#new review image matching

for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/deep_T_PJT__20180316/review'):
    category = root.split('/')[-1]
    for fname in files:
        a=1
        #os.rename(root + '/' + fname, root + '/' + content[category][fname.split('.png')[0]] + '.png')
        scan_path = '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/deep_T_PJT__20180316/scan/' + fname.split('.jpg')[0]+'.png'
        if os.path.isfile(scan_path):
            shutil.copy(scan_path, '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/deep_T_PJT__20180316/matched_scan')
        else:
            print('No matching')

for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/deep_T_PJT__20180316/matched_scan'):
    category = root.split('/')[-1]
    for fname in files:
        a = 1
        # os.rename(root + '/' + fname, root + '/' + content[category][fname.split('.png')[0]] + '.png')
        review_path = '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/deep_T_PJT__20180316/review/' + \
                    fname.split('.png')[0] + '.jpg'
        if os.path.isfile(review_path):
            shutil.copy(review_path,
                        '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/deep_T_PJT__20180316/matched_review')
        else:
            print('No matching')
'''


#f = open('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/deep_T_PJT__20180316/matched_review_120_TESTLR/ALL/ls.txt',encoding="ISO-8859-1")

TESTLR_FILE_NAME = []
for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/test/Review_120_TESTLR'):
    category = root.split('/')[-1]
    for fname in files:
        TESTLR_FILE_NAME.append(fname)


for root, dirs, files in os.walk('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/AOI_Trace/Review_720_HR'):
    category = root.split('/')[-1]
    for fname in files:
        if fname in TESTLR_FILE_NAME:
            shutil.move(root+'/'+fname,
                        '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/AOI_Trace/Review_720_TESTHR/' + fname)


