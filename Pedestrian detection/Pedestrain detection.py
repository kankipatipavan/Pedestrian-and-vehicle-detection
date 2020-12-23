#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2


# In[4]:


import os


# In[5]:


os.chdir('D:\Spark datasets')


# In[6]:


video_src = 'videosample.MP4'


# In[7]:


cap = cv2.VideoCapture(video_src)


# In[8]:


bike_cascade = cv2.CascadeClassifier('pedestrian.xml')


# In[9]:


while True:
    ret, img = cap.read()
	
    
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bike = bike_cascade.detectMultiScale(gray,1.3,2)

    for(a,b,c,d) in bike:
        cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,210),4)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()


# In[ ]:




