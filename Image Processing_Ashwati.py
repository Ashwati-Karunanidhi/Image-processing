#!/usr/bin/env python
# coding: utf-8

# ### Importing necessary libraries

# In[1]:


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# In[3]:


Achu =os.listdir("C:/Users/Ashwati/OneDrive/Desktop/Enoda Stuffs/PPT 1/Achu")


# In[4]:


Kani =os.listdir("C:/Users/Ashwati/OneDrive/Desktop/Enoda Stuffs/PPT 1/Kani")


# In[5]:


Surya =os.listdir("C:/Users/Ashwati/OneDrive/Desktop/Enoda Stuffs/PPT 1/Surya")


# In[6]:


limit = 10
Achu_images =[None]*limit
j=0

for i in Achu:
    if(j<limit):
        Achu_images[j]=imread("C:/Users/Ashwati/OneDrive/Desktop/Enoda Stuffs/PPT 1/Achu/"+i)
        j+=1
    else:
        break


# In[7]:


limit = 10
Kani_images =[None]*limit
j=0

for i in Kani:
    if(j<limit):
        Kani_images[j]=imread("C:/Users/Ashwati/OneDrive/Desktop/Enoda Stuffs/PPT 1/Kani/"+i)
        j+=1
    else:
        break


# In[8]:


limit = 10
Surya_images =[None]*limit
j=0

for i in Surya:
    if(j<limit):
        Surya_images[j]=imread("C:/Users/Ashwati/OneDrive/Desktop/Enoda Stuffs/PPT 1/Surya/"+i)
        j+=1
    else:
        break


# In[9]:


imshow(Achu_images[1])


# In[10]:


imshow(Kani_images[1])


# In[11]:


imshow(Surya_images[1])


# ### Converting into gray scale

# In[12]:


Achu_gray =[None]*limit
j=0

for i in Achu:
    if(j<limit):
        Achu_gray[j] = rgb2gray(Achu_images[j])
        j+=1
    else:
        break

imshow(Achu_gray[1])


# In[13]:


Kani_gray =[None]*limit
j=0

for i in Kani:
    if(j<limit):
        Kani_gray[j] = rgb2gray(Kani_images[j])
        j+=1
    else:
        break

imshow(Kani_gray[1])


# In[14]:


Surya_gray =[None]*limit
j=0

for i in Surya:
    if(j<limit):
        Surya_gray[j] = rgb2gray(Surya_images[j])
        j+=1
    else:
        break

imshow(Surya_gray[1])


# In[15]:


for j in range (10):
    rk_1=Achu_gray[j]
    Achu_gray[j]


# In[16]:


for j in range (10):
    rk_2=Kani_gray[j]
    Kani_gray[j] 


# In[17]:


for j in range (10):
    rk_3=Surya_gray[j]
    Surya_gray[j] 


# In[18]:


Achu_gray[4].shape


# In[19]:


Kani_gray[4].shape


# In[20]:


Surya_gray[4].shape


# ### Resizing the images

# In[21]:


for j in range(10):
    Achu=Achu_gray[j]
    Achu_gray[j]=resize(Achu,(512,512))

imshow(Achu_gray[7])


# In[22]:


for j in range(10):
    Kani=Kani_gray[j]
    Kani_gray[j]=resize(Kani,(512,512))

imshow(Kani_gray[5])


# In[23]:


for j in range(10):
    Surya=Surya_gray[j]
    Surya_gray[j]=resize(Surya,(512,512))

imshow(Surya_gray[3])


# In[24]:


len_of_images_Achu=len(Achu_gray)
len_of_images_Achu


# In[25]:


len_of_images_Kani=len(Kani_gray)
len_of_images_Kani


# In[26]:


len_of_images_Surya=len(Surya_gray)
len_of_images_Surya


# In[27]:


images_size_Achu=Achu_gray[4].shape
images_size_Achu


# In[28]:


images_size_Kani=Kani_gray[4].shape
images_size_Kani


# In[29]:


images_size_Surya=Surya_gray[4].shape
images_size_Surya


# In[30]:


flatten_size_Achu=images_size_Achu[0]*images_size_Achu[1]
flatten_size_Achu


# In[31]:


flatten_size_Kani=images_size_Kani[0]*images_size_Kani[1]
flatten_size_Kani


# In[32]:


flatten_size_Surya=images_size_Surya[0]*images_size_Surya[1]
flatten_size_Surya


# In[33]:


for i in range(len_of_images_Achu):
    Achu_gray[i]=np.ndarray.flatten(Achu_gray[i]).reshape(flatten_size_Achu,1)


# In[34]:


for i in range(len_of_images_Kani):
    Kani_gray[i]=np.ndarray.flatten(Kani_gray[i]).reshape(flatten_size_Kani,1)


# In[35]:


for i in range(len_of_images_Surya):
    Surya_gray[i]=np.ndarray.flatten(Surya_gray[i]).reshape(flatten_size_Surya,1)


# ### Stack individual Array elements into One Array

# In[36]:


Achu_gray=np.dstack(Achu_gray)

Achu_gray=np.rollaxis(Achu_gray,axis=2,start=0)
Achu_gray.shape


# In[37]:


Kani_gray=np.dstack(Kani_gray)

Kani_gray=np.rollaxis(Kani_gray,axis=2,start=0)
Kani_gray.shape


# In[38]:


Surya_gray=np.dstack(Surya_gray)

Surya_gray=np.rollaxis(Surya_gray,axis=2,start=0)
Surya_gray.shape


# In[39]:


Achu_gray = Achu_gray.reshape(len_of_images_Achu,flatten_size_Achu)
Achu_gray.shape


# In[40]:


Kani_gray = Kani_gray.reshape(len_of_images_Kani,flatten_size_Kani)
Kani_gray.shape


# In[41]:


Surya_gray = Surya_gray.reshape(len_of_images_Surya,flatten_size_Surya)
Surya_gray.shape


# In[42]:


Achu_data=pd.DataFrame(Achu_gray)
Achu_data


# In[43]:


Kani_data=pd.DataFrame(Kani_gray)
Kani_data


# In[44]:


Surya_data=pd.DataFrame(Surya_gray)
Surya_data


# In[45]:


Achu_data["Label"]="Achu"
Achu_data


# In[46]:


Kani_data["Label"]="Kani"
Kani_data


# In[47]:


Surya_data["Label"]="Surya"
Surya_data


# ### Combining individual DataFrames to a Single DataFrame

# In[48]:


Person_1=pd.concat([Achu_data,Kani_data])
People=pd.concat([Person_1,Surya_data])
People


# ### Shuffling the Final DataFrame

# In[49]:


from sklearn.utils import shuffle
People_index=shuffle(People).reset_index()
People_index


# ### Remove index from the Final Dataset

# In[50]:


Human=People_index.drop(['index'],axis=1)
Human


# ### Assigning dependent and independent variables

# In[51]:


x=Human.values[:,:-1]
y=Human.values[:,-1]


# In[52]:


x


# In[53]:


y


# ### Assigning train and test dataset

# In[54]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# ### SVM Algorithm

# In[55]:


from sklearn import svm
clf=svm.SVC()
clf.fit(x_train,y_train)


# ### Image Prediction

# In[56]:


y_pred=clf.predict(x_test)
y_pred


# ### Finding Accuracy

# In[57]:


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


# ### Confusion matrix

# In[58]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:




