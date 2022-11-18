# %%
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# %%
import cv2

# %%
import sys

path1 = sys.argv[1]
path2 = sys.argv[2]
path3 = sys.argv[3]

img1 = cv2.imread(path1)
img2 = cv2.imread(path2)
img3 = cv2.imread(path3)

img1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2 , cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3 , cv2.COLOR_BGR2GRAY)

# %%
print(img1.shape)

# %%
# changing images to float type numpy arrrays

img1 = img1.astype(float)
img2 = img2.astype(float)
img3 = img3.astype(float)

# %%
import math

def gauss(dd , sig):
    return math.exp(-0.5 * dd / (sig**2)) * (1 / (2 * math.pi * sig**2))

# %%
# initiLazing the gaussian kernel
K = 3
sig1 = 1.0
sig2 = 1.0

g_ker = np.zeros([K , K])

for i in range(K):
    for j in range(K):
        dd = (i-K//2)**2 + (j-K//2)**2
        g_ker[i][j] = gauss(dd , sig1)

# %%
def upsampler(d_img , I):

    #  dimensions of the down image
    m = len(d_img)
    n = len(d_img[0])

    #  dimensions of the up image
    M = len(I)
    N = len(I[0])

    # upsampled output
    out = np.zeros([M , N])

    # formula
    # Sp = (1/kp) * ∑ Sq↓ * f(||p↓−q↓||) * g(|| Ip− ̃Iq||)

    for i in range(4*K//2 , M-4*K//2):
        for j in range(4*K//2 , N-4*K//2):

            # calculate the intensity kernel around the "I" image
            in_ker = np.zeros([K , K])

            in_ker = I[i-K//2:i+K//2+1 , j-K//2:j+K//2+1] - I[i][j]

            # replace every element with its gaussian
            # in_ker = math.exp(-0.5 * (in_ker*in_ker) / (sig2**2)) * (1 / (2 * math.pi * sig2**2))
            for x in range(K):
                for y in range(K):
                    in_ker[x][y] = gauss(in_ker[x][y]**2 , sig2)

            # do the element wise multiplication
            d_i = int(i//4.0)
            d_j = int(j//4.0)

            S = d_img[d_i-K//2:d_i+K//2+1 , d_i-K//2:d_i+K//2+1]
            k_p = sum((in_ker * g_ker).reshape([K*K]))

            # out[i][j] = sum((S * in_ker * g_ker).reshape([K*K])) / k_p
            out[i][j] = sum((d_img[d_i-K//2:d_i+K//2+1 , d_j-K//2:d_j+K//2+1] * in_ker * g_ker).reshape(K*K)) / sum((in_ker * g_ker).reshape(K*K))

            if i<7 and j<7:
                print(k_p)

    return out

    

# %%


# %%
out2 = upsampler(img2 , img1)

# %%
# padding out2

out2[0:K//2 , :] = out2[K//2:K//2+1 , :]
out2[: , 0:K//2] = out2[: , K//2:K//2+1]

# %%



# %%
# Image.fromarray(out2).show()
# Image.fromarray(img2).show()
# Image.fromarray(out2).show()

# %%
out3 = upsampler(img3 , img1)

# %%
out3[0:K//2 , :] = out3[K//2:K//2+1 , :]
out3[: , 0:K//2] = out3[: , K//2:K//2+1]

# %%
# Image.fromarray(out3).show()
# Image.fromarray(img3).show()

# %%
M = len(img1)
N = len(img1[0])
# combine and convert back to RGB space
res = np.zeros([M , N , 3])
res[:,:,0] = img1
res[:,:,1] = out2
res[:,:,2] = out3

print(res.shape)
# Image.fromarray(res).show()

# %%
res = res.astype(np.uint8)

# %%
res = cv2.cvtColor(res , cv2.COLOR_YCrCb2RGB)

# %%
# plt.imshow(res)

# %%
# Image.fromarray(res).show()

# %%
# Image.fromarray(img1).show()


cv2.imwrite('flyingelephant.jpg' , res)
