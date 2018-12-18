#coding:utf-8

import time
import os, os.path
import cv2
import keras
import matplotlib
import os, random, shutil 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

#select 25% of inliner picture
inlier_new = '/home/joker/Desktop/5002/Q1/Q1_code/newin/'
inlier = "/home/joker/Desktop/5002/Q1/Q1_code/inlier_train/"
test = "/home/joker/Desktop/5002/Q1/Q1_code/test/"

def moveFile(fileDir,tarDir): 
    pathDir = os.listdir(fileDir) 
    filenumber=len(pathDir) 
    rate=0.25 
    picknumber=int(filenumber*rate) 
    sample = random.sample(pathDir, picknumber)
    for name in sample: 
        shutil.move(fileDir+name, tarDir+name) 
    return
moveFile(inlier,inlier_new)

#generate more outlier picture (double)
def generator(imgname,filename,newpath):

    data_gen = ImageDataGenerator(samplewise_center=False,featurewise_std_normalization = False,samplewise_std_normalization = False,zca_whitening = False,rotation_range = 30,
                                  width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True,)


    img=load_img(filename)
    x = img_to_array(img,data_format="channels_last")   
    x=x.reshape((1,) + x.shape)     

    for batch in data_gen.flow(x,batch_size=1,save_to_dir=newpath,save_prefix=imgname,save_format='jpeg'):
        print (batch.shape)
        break



outlier = '/home/joker/Desktop/5002/Q1/Q1_code/outlier_train'
newpath = '/home/joker/Desktop/5002/Q1/Q1_code/new'

for filename in os.listdir(outlier):
        if '.jpg' in filename:
            imgpath = os.path.join(outlier, filename)
            generator(filename,imgpath,newpath)

for filename in os.listdir(newpath):
    shutil.copy('/home/joker/Desktop/5002/Q1/Q1_code/new/'+filename, '/home/joker/Desktop/5002/Q1/Q1_code/outlier_train/'+filename)

resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(64,64,3))



def preprocess(filenames):
    im=[]
    i=0
    l=[]
    for filename in os.listdir(filenames):
        if '.jpg' in filename:
            imgpath = os.path.join(filenames, filename)
            # Use imread in opencv to read the image
            image = cv2.imread(imgpath)
            if True:
                # Resize it to 224 x 224
                image = cv2.resize(image, (64, 64))

                # Convert it from BGR to RGB so we can plot them later (because openCV reads images as BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Now we add it to our array
                im.append(image)
                images = np.array(im, dtype=np.float32)
                images /= 255
                features=resnet50_model.predict(images)
                features = features.reshape(images.shape[0], -1)
    return features

features=preprocess(outlier)
features2=preprocess(inlier_new)

features=list(features)
features2=list(features2)
l=[]
for i in range(len(features)):
    l.append(1)
for i in range(len(features2)):
    l.append(0)
    features.append(features2[i])

X_train, X_test, y_train, y_test = train_test_split(features, l, test_size=0.3, random_state=50)

test = "/home/joker/Desktop/5002/Q1/Q1_code/test/"
t=preprocess(test)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=6,random_state=50, min_samples_split=2, min_samples_leaf=1)
clf.fit(X_train, y_train)
re=clf.predict(t)
re=pd.DataFrame(re)


re.columns=['Result']
re=pd.DataFrame(re)
re.to_csv('Q1_output.csv',index=True,index_label='ID')

