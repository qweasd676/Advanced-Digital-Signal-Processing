import cv2
import numpy as np
from past.builtins import xrange

class hsv_parmeter :
    lower = np.array([0, 90, 0])  #Red lower limit
    upper = np.array([7, 255, 255])  #Red upper limit
    lower1 = np.array([170, 90, 0])  #Dot lower limit
    upper1 = np.array([190, 255, 255])  #Dot upper limit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    gamma = 0.4



if __name__=='__main__':
    src_img = cv2.imread('test.jpg')
    img_=cv2.imread('./result/swt_denoise.jpg',0)
    # img_=cv2.imread('swt_denosie.jpg',0)

    # img_1=cv2.imread('test12.jpg',0)
    src_img_test = src_img.copy() 
    src_img_test2 = src_img.copy()
    src_img_test[:,:,:] = 0
    src_img_copy = src_img_test.copy()
    src_img_copy1 = src_img_test.copy()


    # eroded = cv2.erode(img_,hsv_parmeter.kernel,iterations = 10)    #erode 
    # dilated = cv2.dilate(eroded,hsv_parmeter.kernel,iterations = 10)
    print(type(img_))
    # pdata_blur = cv2.GaussianBlur(img_,(17,17),0)
    
    ridge_filter = cv2.ximgproc.RidgeDetectionFilter_create()
    ridges = ridge_filter.getRidgeFilteredImage(img_)

    # canny = cv2.Canny(img_, 80, 160)
    # Laplacian_ =  cv2.Laplacian(img_,cv2.CV_16S,ksize = 5)

    # x = cv2.Sobel(img_,cv2.CV_16S,1,0,ksize = 3)
    # y = cv2.Sobel(img_,cv2.CV_16S,0,1,ksize = 3 )
    # absX = cv2.convertScaleAbs(x)   # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    # dst = cv2.addWeighted(absX,0.5,absY,0.5,0)  

    # ridges_blur = cv2.blur(~dst,(3,3))
    # __,ridges_threshold = cv2.threshold(dst,100,255,cv2.THRESH_BINARY)

    # contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE  ,cv2.CHAIN_APPROX_NONE)
    

    ridges_blur = cv2.blur(~ridges,(3,3))
    __,ridges_threshold = cv2.threshold(ridges_blur,100,255,cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(ridges_threshold, cv2.RETR_TREE  ,cv2.CHAIN_APPROX_NONE)
    
    # img_copy = src_img_copy.copy()
    for j in contours:
        area = cv2.contourArea(j)
        # if area > 5000 and area < 10000:
        if area > 5000 and area < 50000:
            # print(j)
            img_copy = src_img_copy.copy()
            cv2.drawContours(src_img,j, -1, (0, 255, 255),5)
            cv2.drawContours(img_copy,j, -1, (255, 255, 255),5)
            cv2.fillPoly(src_img_test, [j], (255,255,255))
            cv2.fillPoly(img_copy, [j], (255,255,255))
            
            # cv2.fillPoly(src_img_test2, [j], (255,255,255))
            # cv2.resize(src_img_test2,(src_img_test2.shape[1],src_img_test2.shape[0]),interpolation = cv2.INTER_NEAREST)
            # print(src_img_test2.shape)
            # cv2.imshow('img',src_img_test2)
            # src_img_test = cv2.blur(src_img_test,(3,3))
            img_copy = cv2.resize(img_copy,(img_copy.shape[1],img_copy.shape[0]),interpolation = cv2.INTER_CUBIC)
            eroded = cv2.erode(img_copy,hsv_parmeter.kernel,iterations = 10)    #erode 
            img_copy = cv2.dilate(eroded,hsv_parmeter.kernel,iterations = 10)

            segmentation =cv2.bitwise_and(src_img_test2 ,src_img_test) 

            segmentation2 =cv2.bitwise_and(src_img_test2 ,img_copy) 


            index = j.argmax(axis = 0)
            index1 = j.argmin(axis = 0)
            data_max = j[index,xrange(j.shape[1])]
            data_min = j[index1,xrange(j.shape[1])]

            # print(data_max,data_min)
            # print(j)
            cv2.imshow('img',src_img_test)
            # cv2.imshow('img1',img_copy)
            cv2.imshow('img2',segmentation)
            if area >= 5000:
                img_test = src_img_test.copy()
                contours1, hierarchy1 = cv2.findContours(img_test[:,:,0], cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                for i in contours1:
                    
                    img_copy_ = img_copy.copy()
                    src_img_copy11 = src_img_copy1.copy()
                    # cv2.drawContours(img_copy_,i,-1, (0, 255, 255),5) 
                    cv2.fillPoly(src_img_copy11, [i], (255,255,255))
                    segmentation11 =cv2.bitwise_and(src_img_test2 ,src_img_copy11)
                       
                    cv2.imshow('img_copy_',img_copy_)
                    cv2.imshow('segmentation11',segmentation11)
                    
                    # cv2.waitKey(0)
            cv2.waitKey(0)



    print(src_img_test2.shape ,src_img_test.shape , img_.shape)
    segmentation =cv2.bitwise_and(src_img_test2 ,src_img_test) 
    segmentation2 =cv2.bitwise_and(src_img_test2 ,img_copy) 

    img1 = src_img_test.copy()
    img1[:,:,0] = img_
    img1[:,:,1] = img_
    img1[:,:,2] = img_
    segmentation1 = cv2.bitwise_and(img1,~src_img_test)


    # cv2.imshow('ds1a',src_img_test)
    cv2.imshow('src',~src_img_test)
    cv2.imshow('segmentation',segmentation)
    cv2.imshow('segmentation2',segmentation2)
    cv2.imshow('ridges_white',~ridges)
    cv2.imshow('ridges',ridges)
    cv2.imwrite('./result/segmentation/wavelet_result.jpg',segmentation)
    # cv2.imwrite('./result/gray_result.jpg',segmentation)

    cv2.imwrite('ridges.jpg',ridges)
    cv2.imwrite('segmentation.jpg',segmentation)
    cv2.imwrite('segmentation1.jpg',segmentation1)

    cv2.imwrite('src_img_test.jpg',src_img_test)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()