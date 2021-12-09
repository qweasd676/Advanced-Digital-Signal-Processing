#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse 
from sklearn import  linear_model
from sklearn.preprocessing import  PolynomialFeatures


# In[8]:


def Polynomial(S, H, T):
    minX = min(S)
    maxX = max(S)
    print(min(S), min(H), max(S), max(H))
    X = np.arange(0, 255).reshape([-1,1])
    poly_reg = PolynomialFeatures(degree = 3)
    X_poly = poly_reg.fit_transform(S)
    lin_reg_2 = linear_model.LinearRegression(fit_intercept = False)
    lin_reg_2.fit(X_poly, H)
    
    print('Coefficients:',lin_reg_2.coef_)
    #查看回歸方程截距
    print('intercept',lin_reg_2.intercept_)
    plt.xlim(0, 255)
    plt.ylim(0, 360)
    plt.xlabel("S")
    plt.ylabel("H")
    plt.title(T)
    plt.scatter(S, H, color='red')
    plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
    plt.savefig(T + ".png")
    plt.show()
    
    with open("./" + T + ".txt", "w") as f:
        f.write("Min S:" + str(min(S)[0]) + "\n")
        f.write("Max S:" + str(max(S)[0]) + "\n")
        f.write("Min H:" + str(min(H)[0]) + "\n")
        f.write("Max H:" + str(max(H)[0]) + "\n")
        f.write("Coef:" + str(lin_reg_2.coef_[0]))

# In[3]:


def cal_hsv(image, top, down):
    global All_S, All_H
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    S = hsv[top[1]:down[1], top[0]:down[0], 1].reshape(-1,1)
    H = hsv[top[1]:down[1], top[0]:down[0], 0].reshape(-1,1)
    All_S.append(S)
    All_H.append(H)


# In[4]:


def mouse_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))
        
    if len(points) == 2:
        img = ori_img.copy()
        cv2.rectangle(img, points[0], points[1], (255,255,0), 2)
        cal_hsv(img, points[0], points[1])
        cv2.imwrite("strawberry.png", img)
        cv2.imshow("image",img)
        points = []


# In[7]:


def ARGS():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", '-p', help="Image file's folder path", type=str, required=True)
    return parser.parse_args()

# In[5]:


if __name__ == "__main__":
    All_S = []
    All_H = []
    points = []
    args = ARGS()
    folder_path = args.path
    Images = os.listdir(folder_path)
    for i in Images:
        if i[-3:] != "jpg":
            continue
        img = cv2.imread(os.path.join(folder_path, i))
        ori_img = img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_event)
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    flatten_list_S = [i for item in All_S for i in item]
    flatten_list_H = [i for item in All_H for i in item]
    NP_ALL_S = np.asarray(flatten_list_S)
    NP_ALL_H = np.asarray(flatten_list_H)
    
    Polynomial(NP_ALL_S, NP_ALL_H, "All")
    
    SL = np.delete(NP_ALL_S, np.where(NP_ALL_H > 50)).reshape(-1,1)
    HL = np.delete(NP_ALL_H, np.where(NP_ALL_H > 50)).reshape(-1,1)

    SH = np.delete(NP_ALL_S, np.where(NP_ALL_H < 150)).reshape(-1,1)
    HH = np.delete(NP_ALL_H, np.where(NP_ALL_H < 150)).reshape(-1,1)
    
    print("test",SL)
    
    
    if SL.size > 0:
        Polynomial(SL, HL, "HSV_analysis_Low")
    if SH.size > 0:
        Polynomial(SH, HH, "HSV_analysis_High")

