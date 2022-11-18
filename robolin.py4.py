# %%
from PIL import Image, ImageOps, ImageFilter

# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math

# %%
import cv2
import sys

# %%
path = sys.argv[1]

op_path = path[0:2] + 'robolin-' + path[2:]

# %%
img1 = Image.open(path)
img0 = np.array(img1)
img1 = ImageOps.grayscale(img1)
img1 = img1.filter(ImageFilter.GaussianBlur(2.0))

# converting to numpy array
img1 = np.array(img1)
plt.imshow(img1)

m = len(img1)
n = len(img1[0])


# %%
# smoothing by gaussian
 

# %%
# lets take laplacian

L = [
    [0 , 1, 0],
    [1, -4, 1],
    [0, 1, 0]
]
def laplacian(img1):
    img2 = np.zeros([m , n])

    for i in range(1,m-1):
        for j in range(1,n-1):
            img2[i][j] = img1[i][j+1]+img1[i+1][j]+img1[i-1][j]+img1[i][j-1]+img1[i+1][j+1]+img1[i-1][j+1]+img1[i-1][j-1]+img1[i+1][j-1]- 8*img1[i][j]
    
    return img2


# %%
f_2 = laplacian(img1)

# plt.imshow(f_2)

# %%
# first derivative

# in x
f_x = np.zeros([m , n])
f_y = np.zeros([m , n])

for i in range(1, m-1):
    for j in range(1 , n-1):
        f_x[i][j] = abs(img1[i+1][j]/2.0 - img1[i-1][j]/2.0)
        f_y[i][j] = abs(img1[i][j+1]/2.0 - img1[i][j-1]/2.0)

# %%
# edge map generation

M = np.zeros([m , n])

for i in range(m):
    for j in range(n):
        M[i][j] = math.sqrt(f_x[i][j]**2 + f_y[i][j]**2)


# %%
# plt.imshow(M)

# %%
# do thresholding

max = 0

for i in range(m):
    for j in range(n):
        if max <= M[i][j]:
            max = M[i][j]

print(max)

# %%
# adding canny detection for thin edges

M = M.astype(np.uint8)
bin_img = cv2.Canny(img0 , 30, 150, apertureSize=3)
 

# %%
# hoiugh transform

# bin_img = np.zeros([m , n])
# for i in range(m):
#     for j in range(n):
#         if M[i][j] >= 100.0:
#             bin_img[i][j] = 255
#         else:
#             bin_img[i][j] = 0

# bin_img = cv2.threshold(M , 127, 255, cv2.THRESH_BINARY)
# bin_img = np.array(bin_img)

bin_img = bin_img.astype(np.uint8)

H_img = cv2.HoughLines(bin_img , 1 , np.pi/180 , 100)
H_img = np.array(H_img)
# H_img.show()

# %%
print(H_img.shape)

# %%
# drawing lines on img0

for line in H_img:
    r , th = np.array(line[0] , dtype=np.float64)

    cos = np.cos(th)
    sin = np.sin(th)
    dis = 1200

    x_1 = int(r*cos - dis*sin)
    x_2 = int(r*cos + dis*sin)
    y_1 = int(r*sin + dis*cos)
    y_2 = int(r*sin - dis*cos)

    cv2.line(img0 , (x_2,y_2) , (x_1,y_1) , (255 , 0 , 0) , 2)

# %%
# plt.imshow(img0)

cv2.imwrite(op_path , img0)

# %%
# select a space for rho and theta

print((m**2 + n**2)**0.5)
rho = np.arange(0, 828.0 , 4)

theta = np.arange(0 , np.pi , 0.04)

print(theta)

# %%
p_max = 828.0

theta_max = np.pi

hough = np.zeros([len(rho) , len(theta)])

print(hough.shape)

# %%
count = 0

for x in range(m):
    for y in range(n):
        if M[x][y] >= 200.0:
            count += 1

            for p in rho:
                for th in theta:
                    if abs(x * math.cos(th) + y*math.sin(th) - p) <= 0.1:
                        hough[int(p / 4)][int(th / 0.04)] += 1

print(count)

# %%
print(hough[50:56,0])

# %%
# detecting lines using hough matrix

# plt.imshow(hough)

# %%
# Lets get the rho and theta corres to lines

sol = []

for p in rho:
    for th in theta:
        if th>0 and th<np.pi and p>0 and p<p_max:
            p = int(p)
            th = int(th)
            if hough[int(p)][int(th)]>hough[int(p-4)][int(th-0.04)] and hough[int(p)][int(th)]>hough[int(p-4)][int(th-0.04)] and hough[p][th]>hough[p-10][th+1] and hough[p][th]>hough[p+10][th+1]:
                sol.append([p , th])

# %%
print(len(sol))

res = np.zeros([m , n])
for line in sol:

    _rho = line[0]
    _th = line[1]

    for x in range(m):
        for y in range(n):
            if abs(x*math.cos(_th*math.pi/180.0) + y*math.sin(_th*math.pi/180.0) - _rho) <= 2.0:
                res[x][y] = 255

# plt.imshow(res)


