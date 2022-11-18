from ctypes import sizeof
import cv2
import numpy as np
import sys

# problem specs
height = 300
width = 500
radius = 25

# a two digit number
num = sys.argv[1]

# center coordinate arrays
# x-axis offset between the two digits
offset = 80+3*(50)+2*(8)

y = [5+25]
for i in range(4):
    y.append(y[-1] + 58)

x = [36+25]
for i in range(2):
    x.append(x[-1] + 58)

# draws a 25px circle at x,y
def put_circle(x , y , img , radius):
    center = (x,y)
    img = cv2.circle(img , center, radius, [255] , thickness=-radius)

def draw_0(offset , img):
    print("drawinfg zero")
    for c_2 in y:
        put_circle(x[0]+offset , c_2, img, radius)
        put_circle(x[2]+offset , c_2, img, radius)
    
    put_circle(x[1]+offset,y[0] , img, radius)
    put_circle(x[1]+offset, y[4], img, radius)

def draw_1(offset , img):
    print("drawing one")
    
    for c_2 in y:
        put_circle(x[2]+offset , c_2 , img, radius)

def draw_2(offset , img):
    print("drawing two")
    for c_1 in x:
        # filling rows 1,3 and 5
        put_circle(c_1+offset, y[0], img,radius)
        put_circle(c_1+offset, y[2], img,radius)
        put_circle(c_1+offset, y[4], img,radius)
    
    put_circle(x[2]+offset,y[1],img,radius)
    put_circle(x[0]+offset, y[3], img, radius)

def draw_3(offset , img):
    print("drawing three")
    for c_1 in x:
        # filling rows 1,3 and 5
        put_circle(c_1+offset, y[0], img,radius)
        put_circle(c_1+offset, y[2], img,radius)
        put_circle(c_1+offset, y[4], img,radius)
    
    put_circle(x[2]+offset,y[1],img,radius)
    put_circle(x[2]+offset, y[3], img, radius)

def draw_4(offset , img):
    print("drawing four")
    for c_1 in x:
        put_circle(c_1+offset, y[2], img, radius)
    
    for c_2 in y:
        put_circle(x[2]+offset, c_2, img, radius)
    
    for c_2 in range(3):
        put_circle(x[0]+offset, y[c_2], img, radius)

def draw_5(offset, img):
    print("drawing five")
    for c_1 in x:
        # filling rows 1,3 and 5
        put_circle(c_1+offset, y[0], img,radius)
        put_circle(c_1+offset, y[2], img,radius)
        put_circle(c_1+offset, y[4], img,radius)
    
    put_circle(x[2]+offset,y[3],img,radius)
    put_circle(x[0]+offset, y[1], img, radius)

def draw_6(offset, img):
    print("drawing six")
    for c_1 in x:
        # filling rows 1,3 and 5
        put_circle(c_1+offset, y[0], img,radius)
        put_circle(c_1+offset, y[2], img,radius)
        put_circle(c_1+offset, y[4], img,radius)
    
    put_circle(x[2]+offset,y[3],img,radius)
    put_circle(x[0]+offset, y[1], img, radius)
    put_circle(x[0]+offset,y[3],img,radius)

def draw_7(offset , img):
    print("drawing seven")
    
    for c_2 in y:
        put_circle(x[2]+offset , c_2 , img, radius)

    for c_1 in x:
        put_circle(c_1+offset, y[0], img, radius)

def draw_8(offset, img):
    print("drawing eight")
    for c_1 in x:
        # filling rows 1,3 and 5
        put_circle(c_1+offset, y[0], img,radius)
        put_circle(c_1+offset, y[2], img,radius)
        put_circle(c_1+offset, y[4], img,radius)
    
    put_circle(x[2]+offset,y[3],img,radius)
    put_circle(x[0]+offset, y[1], img, radius)
    put_circle(x[0]+offset,y[3],img,radius)
    put_circle(x[2]+offset,y[1],img,radius)

def draw_9(offset, img):
    print("drawing nine")
    for c_1 in x:
        # filling rows 1,3 and 5
        put_circle(c_1+offset, y[0], img,radius)
        put_circle(c_1+offset, y[2], img,radius)
        put_circle(c_1+offset, y[4], img,radius)
    
    put_circle(x[2]+offset,y[3],img,radius)
    put_circle(x[0]+offset, y[1], img, radius)
    put_circle(x[2]+offset,y[1],img,radius)

img = np.zeros(shape=[height, width, 1], dtype=np.uint8)

# actually drawing the digits
# draw_5(0 , img)
# draw_7(offset , img)

c1 = num[0]
if c1=='0':
    draw_0(0, img)
if c1=='1':
    draw_1(0, img)
if c1=='2':
    draw_2(0, img)
if c1=='3':
    draw_3(0, img)
if c1=='4':
    draw_4(0, img)
if c1=='5':
    draw_5(0, img)
if c1=='6':
    draw_6(0, img)
if c1=='7':
    draw_7(0, img)
if c1=='8':
    draw_8(0, img)
if c1=='9':
    draw_9(0, img)

c2 = num[1]
# print(len(c2))
if c2=='0':
    draw_0(offset, img)
if c2=='1':
    draw_1(offset, img)
if c2=='2':
    draw_2(offset, img)
if c2=='3':
    draw_3(offset, img)
if c2=='4':
    draw_4(offset, img)
if c2=='5':
    draw_5(offset, img)
if c2=='6':
    draw_6(offset, img)
if c2=='7':
    draw_7(offset, img)
if c2=='8':
    draw_8(offset, img)
if c2=='9':
    draw_9(offset, img)

cv2.imwrite("./dotmatrix.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

