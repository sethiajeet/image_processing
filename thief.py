# %%
from PIL import Image , ImageOps, ImageFilter
import cv2
import sys

# %%
print(255 / 55.0)

# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import string

# %%
path = sys.argv[1]

op_path = path[0:2] + 'enhanced-' + path[2:]

print(op_path)

# %%
img1 = Image.open(path)
img1.show()
img1 = ImageOps.grayscale(img1)

img_b = np.array(img1)
# img_r = img1[:,:,0]..
# img_g = img1[:,:,1]
# img_b = img1[:,:,2]

plt.imshow(img_b)

# %%
m = len(img_b)
n = len(img_b[0])

print(m)
print(n)

# %%
# lets do log corrcetion
gam_img = np.zeros([m , n])
c = 45.0

for i in range(m):
    for j in range(n):
        gam_img[i][j] = c*math.log(1 + img_b[i][j])

gam_img = Image.fromarray(gam_img)
# gam_img.show()

gam_img = np.array(gam_img)
print(gam_img.shape)

# %%
hist = np.zeros(256)

for i in range(m):
    for j in range(n):
        hist[int(gam_img[i][j])] += 1 

print(hist[0:10])

# %%
L_gamma = 0

for i in range(m):
    for j in range(n):
        if L_gamma < gam_img[i][j]:
            L_gamma = int(gam_img[i][j])

# %%
cdf = np.zeros(256)
cdf[0] = hist[0]

cdf_min = hist[0]
cdf_max = hist[0]

for j in range(1,256):
    cdf[j] = cdf[j-1] + hist[j]

    if cdf_min > cdf[j]:
        cdf_min = cdf[j]
    
    if cdf_max < cdf[j]:
        cdf_max = cdf[j]

print(cdf_min)


# %%
# doing histogram equalization for blue component

op = np.zeros([m , n])

for i in range(m):
    for j in range(n):
        op[i][j] = (cdf[int(gam_img[i][j])] - cdf_min) * L_gamma / (m*n - cdf_min)

plt.imshow(op)

# %%
# processing op array
max = 0
op1 = np.zeros([m , n])

for i in range(m):
    for j in range(n):
        op1[i][j] = int(op[i][j])
        if max <= op1[i][j]:
            max = op[i][j]

op1.astype(np.uint8)
print(max)
print(type(op1[0][0]))

# %% [markdown]
# 

# %%
# op.astype(np.uint8)
# res_img = Image.fromarray(op)

# # for c in path:
# #     if c.isdigit():
# #         key = '../enhanced-cctv' + c + '.jpg'
# res_img.convert('L')

# res_img.show()

# res_img.save('../enh-1.jpg')
cv2.imwrite(op_path , op)



