# %%
import time

start = time.time()



# %%
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

# %%
import cv2

# %%
import sys

# %%

path = sys.argv[1]
img1 = cv2.imread(path)



# %%
img1 = np.array(img1)

plt.imshow(img1)

# %%
img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)

img1_v = img1_hsv[:,:,2]

# %%
plt.imshow(img1_v)

# print(img1_hsv.shape)

# %%
# lets apply median filter
m = len(img1)
n = len(img1[0])

print(m)
print(n)

# %%
img_fin = np.zeros([m , n])

# %%
K = 11

# %%
for i in range(K//2 , m-K//2):
    for j in range(K//2 , n-K//2):
        
        med_array = img1_v[i-K//2:i+K//2+1 , j-K//2:j+K//2+1].reshape([K*K])
        
        # diagonal array
        
        # med1 = np.median(m1)
        # med2 = np.median(m2)
        # res = np.array([med1 , med2 , img1_v[i][j]])

        img_fin[i][j] = np.median(med_array)

# %%
# Image.fromarray(img_fin).show()
# Image.fromarray(img1_v).show()

# %%
diff_img = img1_v - img_fin
# Image.fromarray(diff_img).show()

# %%
img_fin1 = img_fin + 0.2*(img1_v-img_fin)
# Image.fromarray(img_fin1).show()
# Image.fromarray(img1_v).show()

# %%
img1_hsv[:,:,2] = img_fin1

output = cv2.cvtColor(img1_hsv , cv2.COLOR_HSV2BGR)
img1 = cv2.cvtColor(img1 , cv2.COLOR_BGR2RGB)

# Image.fromarray(output).show()
# Image.fromarray(img1).show()

# %%

cv2.imwrite('./denoised.jpg' , output)


# print(23*2.3)

end = time.time()
print(end - start)