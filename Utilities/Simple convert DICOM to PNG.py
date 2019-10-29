import cv2
import os
import pydicom
import shutil

print(os.listdir("../input"))
inputdir = '../input/siim-acr-pneumothorax-segmentation/stage_2_images/'
outdir = './images/'
os.mkdir(outdir)

test_list = [ f for f in  os.listdir(inputdir)]
i = 0
for f in test_list:   # remove "[:10]" to convert all images 
    ds = pydicom.read_file(inputdir + f) # read dicom image
    img = ds.pixel_array # get image array
    cv2.imwrite(outdir + f.replace('.dcm','.png'),img) # write png image
    print("image :", i)
    i+=1
    
    





# Create a new zipfile ( I called it myfile )
shutil.make_archive("./stage2_test_images", 'zip', "./images")
shutil.rmtree('images')    
os.listdir("./")
