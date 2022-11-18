# %%
from PIL import Image , ImageOps, ImageFilter
from scipy.fft import ifft2
from scipy.fft import fft2 , fftshift

# %%
import numpy as np
import matplotlib.pyplot as plt
import math

# %%
import cv2

# %%
import sys

# %%
path = sys.argv[1]

op_path = './cleaned-gutter.jpg'

# %%
img1 = Image.open(path)
# img1 = ImageOps.grayscale(img1)

# img3 = img1.filter(ImageFilter.BoxBlur(70))
# plt.imshow(img3)

img1 = np.array(img1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
m = len(img1)
n = len(img1[0])

# plt.imshow(img1)
# print(img1.shape)

# %%
h_img,s_img,v_img = cv2.split(img1)
# plt.imshow(v_img)

# %%
ft_1 = fftshift(fft2(v_img))
ft_2 = np.zeros([m , n])

for i in range(m):
    for j in range(n):
        
        ft_2[i][j] = ft_1[i][j]

# %%
print(ft_2[100][200])

# %%
# plt.imshow(ft_2)

ft_2 = Image.fromarray(ft_2)
# ft_2.show()

# %%
# generating a smooth mask

mask = np.zeros([m , n])

dis = n / 6.0
inner = 5.0

for i in range(m):
    for j in range(n):

        curr = (i-m/2.0)**2 + (j-n/2.0)**2
        if curr <= dis**2:
            mask[i][j] = 1
        # mask[i][j] *= 255.0

# gaussianising the mask 
# -((i-m/2.0)**2 + (j-n/2.0)**2) / sig**2

g_mask = np.ones([m , n])
sig = 20.0

for i in range(m):
    for j in range(n):
        g_mask[i][j] = math.exp(-((i-m/2.0)**2 + (j-n/2.0)**2) / sig**2) * mask[i][j]

# g_mask = Image.fromarray(g_mask).show()
# print(mask[0][0])

# %%
# doing a high pass filter
# ft_2 = np.array(ft_2)
# g_mask = np.array(g_mask)

for i in range(m):
    for j in range(n):
        ft_1[i][j] *= g_mask[i][j]

# ft_2 = Image.fromarray(ft_2)
# ft_1.show()

# %%
sol = ifft2(ft_1)

res = np.zeros([m , n])
for i in range(m):
    for j in range(n):
        
        res[i][j] = float(abs(sol[i][j]))
print(res[0][1])

# %%
res = Image.fromarray(res)

# res.show()

# %%
final = v_img - res

# Image.fromarray(final).show()

# %%
for i in range(m):
    for j in range(n):
        if final[i][j] <= -10:
            final[i][j] = 0
        else:
            final[i][j] = 255

# Image.fromarray(final).show()
cv2.imwrite(op_path , final)

