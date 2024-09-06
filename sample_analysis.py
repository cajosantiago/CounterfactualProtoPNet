import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-mskdir', nargs=1, type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test set to be analyzed
test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
#test_mask_dir = args.mskdir[0]

# load the model

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
load_model_name = args.model[0] #'10_18push0.7822.pth'

import subprocess
for i in os.listdir(test_image_dir+"0/"):
    p = subprocess.Popen("python local_analysis.py -modeldir " + load_model_dir
                     +" -model " + load_model_name
                     +" -imgdir " + test_image_dir+"0/"
                     +" -img " + i
                     +" -imgclass 0"
                     , shell=True)
    p.wait()
for i in os.listdir(test_image_dir+"1/"):
    p = subprocess.Popen("python local_analysis.py -modeldir " + load_model_dir
                     +" -model " + load_model_name
                     +" -imgdir " + test_image_dir+"1/"
                     +" -img " + i
                     +" -imgclass 1"
                     , shell=True)
    p.wait()