# %%
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# %%
import cv2

# %%
import sys

path = sys.argv[1]
img1 = cv2.imread(path)

# %%
plt.imshow(img1)

# %%
import math

# %%
img1 = img1.astype(float)

# %%
# lets calculate mean R-G , G-B , G-R , 2G-R-B

m = len(img1)
n = len(img1[0])

r_g = 0
b_g = 0
r_b = 0
grb = 0

for i in range(m):
    for j in range(n):
        r_g += abs(img1[i][j][0] - img1[i][j][1]) / (1.0 * m * n)
        b_g += abs(img1[i][j][2] - img1[i][j][1]) / (1.0 * m * n)
        r_b += abs(img1[i][j][0] - img1[i][j][2]) / (1.0 * m * n)
        grb += abs(2*img1[i][j][1] - img1[i][j][0] - img1[i][j][2]) / (1.0 * m * n)


# %%
# print(r_g)
# print(r_b)
# print(b_g)
# print(grb)


# %%
# building=1, grass=2 and road=3

ans = 0

if(grb >= 12):
    ans = 2

elif(b_g < 6):
    ans = 1

else:
    ans = 3


# %%
print(ans)


