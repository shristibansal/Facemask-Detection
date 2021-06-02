import numpy as np
import os
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow as tf
import cv2

model = load_model(r'C:\Users\Khushi Bansal\Desktop\class\my projects\facemask\weights\face.hdf5')

face_clf = cv2.CascadeClassifier(r'C:\Users\Khushi Bansal\Desktop\class\my projects\facemask\haarcascades\haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

labels = {0:'WITHOUT MASK',1:'MASK'}

color_dict = {0:(0,0,255),1:(0,255,0)}

#code
while (True):
    ret,frame = source.read() 
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_clf.detectMultiScale(grey,1.3,5)
    
    for x,y,w,h in faces:
        faceImg = grey[y:y+w, x:x+w]
        resized = cv2.resize(faceImg,(150,150))
        normalized = resized/255.0
        reshaped = np.reshape(normalized,(1,150,150,3)) #error
        
        result = model.predict(reshaped)[0]
        
      #  label = np.argmax(result,-1)
        
        print(result[0])
        
        if result[0] > 0.50:
            label = 0
        else:
            label = 1
        
        print(label)
        
        cv2.rectangle(frame, (x,y),(x+w,y+h), color_dict[label], 2)
        
        cv2.rectangle(frame, (x,y-50),(x+w,y), color_dict[label], -1)
        
        cv2.putText(frame,labels[label],(x,y-10), cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
    cv2.imshow('MASK DETECTION', frame)
    
    key = cv2.waitKey(1)
    if key == 27 :
        break
    
cv2.destroyAllWindows()
source.release()
        
    
    
    
    
    
    
    
 