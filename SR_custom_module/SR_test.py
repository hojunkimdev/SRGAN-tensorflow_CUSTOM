import SuperResolution

input_dir_LR = './data/inference/test/LR'
output_dir = './data/input/verX_120_LR'
checkpoint = './experiment_SRGAN_VGG54/ver7_shTrainSet_train/model-40000'

sr = SuperResolution.SR(input_dir_LR, output_dir, checkpoint)


sr.inference()