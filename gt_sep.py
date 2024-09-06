from PIL import Image, ImageDraw
import numpy as np
from pandas import read_csv,read_excel
import os

directory = './../CBIS/images/train_augmented/1/'
directory2 = './../CBIS/images/training_augmented/1/'
directory3 = './../CBIS/masks_augmented/'

for filename in os.listdir(directory):
    image = Image.open(directory+filename)
    imagename = filename.split("_")
    #print(filename)
    
    if  imagename[0] == "1" or imagename[0] == "0":
        image.save(directory2+imagename[2]+imagename[3]+imagename[4]+imagename[5]+imagename[6]+imagename[7]+'.jpg')
    else:
        image.save(directory3+imagename[4]+imagename[5]+imagename[6]+imagename[7]+imagename[8]+imagename[9]+'.jpg')