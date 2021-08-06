import os
from glob import glob

'''
A_files_dir_path = '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/ver11_cycleGAN_train/Review_120_cycle_LR'
B_files_dir_path = '/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/ver11_cycleGAN_train/Review_480_HR'
A_files = glob(A_files_dir_path + '/*.*')
B_files = glob(B_files_dir_path + '/*.*')


index_root_path = './'
index_path = os.path.join(index_root_path, 'HTML_comparison_index.html')
index = open(index_path, "w")
index.write("<html><body><table><tr>")
index.write("<th>A name</th><th>B name</th><th>A image</th><th>B image</th></tr>")


for i in range(0,len(A_files)):
    A_path = os.path.join(A_files_dir_path, '/{0}'.format(os.path.basename(A_files[i])))
    B_path = os.path.join(B_files_dir_path, '/{0}'.format(os.path.basename(B_files[i])))
    index.write("<td>%s</td>" % os.path.basename(A_path))
    index.write("<td>%s</td>" % os.path.basename(B_path))
    index.write("<td><img src='%s'></td>" % (A_files[i] if os.path.isabs(A_files[i]) else (
            '..' + os.path.sep + A_files[i])))
    index.write("<td><img src='%s'></td>" % (B_files[i] if os.path.isabs(B_files[i]) else (
            '..' + os.path.sep + B_files[i])))
    index.write("</tr>")
index.close()

'''

index_root_path = './'
index_path = os.path.join(index_root_path, 'HTML_comparison_index.html')
index = open(index_path, "w")
index.write("<html><body><table><tr>")

files_dir_list = []
files_dir_list.append('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/Review_120_fns_LR')
files_dir_list.append('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/ver11_cycleGAN_train/Review_120_cycle_LR')
files_dir_list.append('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/ver11_cycleGAN_train/Review_480_HR')
files_dir_list.append('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/inference/ver11_cycleGAN_SR')
files_dir_list.append('/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/Train_Val_ksh/scan/crop/Train/ALL')


files_list = []

for files_dir in files_dir_list:
    files_list.append(glob(files_dir + '/*.*'))
    print("path: {}".format(files_dir))

    if files_dir=='/home/dl2/Desktop/workspace/khj/SDC/SRGAN-tensorflow-master/data/train/SH_trainSet/Train_Val_ksh/scan/crop/Train/ALL':
        # Rename Th
        index.write("<th>Scan_120 image</th>")
    else:
        index.write("<th>{} </th>".format(files_dir.split('/')[-1]))


for i in range(0,len(files_list[0])):
    index.write("<tr>")
    for j in range(0,len(files_list)):
        index.write("<td><img src='%s'></td>" % (files_list[j][i]))
        #print("i = {0}, j={1}".format(i,j))

    index.write("</tr>")

for files in files_list:
    print("dir size = {}".format(len(files)))

index.close()
