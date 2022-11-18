#!/usr/bin/env python
# coding: utf-8

# In[38]:


import PIL
import sys

# In[39]:


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# In[40]:
path = sys.argv[1]

img = Image.open(path)
img = np.array(img)


# In[41]:


plt.imshow(img)


# In[42]:


print(len(img))
print(len(img[0]))


# In[43]:


x = 199
y = 189

i1 = img[0:x , 0:y].copy()
i2 = img[x:x+210 , 0:y].copy()

plt.imshow(i1)


# In[44]:


plt.imshow(i2)


# In[45]:


# rotating i2 and pasting it correctly

m2 = len(i2)
n2 = len(i2[2])

for i in range(m2):
    img[i,0:n2] = i2[m2-1-i]

plt.imshow(img)


# In[46]:


m1 = 199

# fixing colors for i1
rows = len(i1)
cols = len(i1[0])
print(rows)
print(cols)

# lets try RBG ( it worked :) )
for i in range(rows):
    for j in range(cols):
        temp = i1[i][j][1]
        i1[i][j][1] = i1[i][j][2]
        i1[i][j][2] = temp
        
# placing i1 at its location
for i in range(m2 , m1+m2):
    img[i-10,:y]= i1[i-m2]

plt.imshow(i1)


# In[47]:


plt.imshow(img)


# In[50]:


# flipping i3 left to right
x1 = 149
x2 = 329
y1 = 514
y2 = 699
length = y2-y1

i3 = img[x1:x2 , y1:y2].copy()

# for i in range(length):
#     i3[:,i] = i3[:,length-1-i]

# rotating i2 and pasting it correctly

m3 = len(i3)
n3 = len(i3[0])

for i in range(length):
    img[x1:x2 , y1+i] = i3[: , length-1-i]

plt.imshow(img)
# plt.imshow(i3)


# In[54]:


# rotatting the last strip
M = 421
N = 797

x4 = 369
y4 = 369

i4 = img[x4:M,y4:N].copy()
# i4_len = len()


for i in range(x4,M):
    img[i][y4:N] = i4[len(i4)-1-i+x4]


# In[55]:


# removing that portion from image and padding it
# img[370:M,376:N,0:3] = 255
plt.imshow(img)


# In[56]:


# filling the gap with padding
gap = img[399:409,0:189]

for i in range(len(gap)):
    gap[i] = img[399-i][0:189]

img = Image.fromarray(img)
img = img.save('./jigsolved.jpg')

# plt.imshow()
# In[ ]:




