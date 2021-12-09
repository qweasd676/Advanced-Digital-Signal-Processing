#!/usr/bin/env python
# coding: utf-8
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import pywt
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis

from skimage.restoration import (denoise_wavelet,estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import skimage.io
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio as PSNR
import math

class hsv_parmeter :
    lower = np.array([0, 90, 0])  #Red lower limit
    upper = np.array([7, 255, 255])  #Red upper limit
    lower1 = np.array([170, 90, 0])  #Dot lower limit
    upper1 = np.array([190, 255, 255])  #Dot upper limit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    gamma = 0.4


# image pre-processing
#region
def Histogram_Equalization(img):
    # 使用函数计算
    Imin, Imax = cv2.minMaxLoc(img)[:2]
    # 使用numpy计算
    # Imax = np.max(img)
    # Imin = np.min(img)
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * img + b
    out = out.astype(np.uint8)  
    
    return out


# In[3]:


def Gamma_transformation(img,gamma):
    img_gamma = np.power(img/float(np.max(img)), gamma)
    img_gamma = (img_gamma *255). astype(int)
#     plt.figure("Image") # 图像窗口名称   matplotlib.pyplot.imshow()   
#     plt.imshow(img_gamma,'gray')
#     plt.axis('off') # 关掉坐标轴为 off
#     plt.title('image') # 图像题目
#     plt.show() 
    cv2.imwrite('img_gamma.jpg',img_gamma) 
    img_gamma = img_gamma.astype('uint8')
    #   解決通道問題
    
    return img_gamma


# In[4]:


def linear_transformation(a , img):
    """
    線性變換
    解釋：
    假設輸入圖片為I，寬為w，高為h，輸出影象記為O，影象的線性變換公式為
                                               O(r,c) = a*I(r,c)+b,    0<=r<h, o<=c<w
    a為係數。a>1時，影象對比度被放大，0<a<1時影象對比度被縮小。常數項b用於調節亮度，b>0時亮度增強，b<0時對比度降低。
    """
    O = float(a) * img
    O[O>255] = 255
    O = np.round(O)
    O = O.astype(np.uint8)
    return O

# In[5]:


def hsv_transformation(pdata_,lower,upper):

    # change to hsv model
    hsv_pdata = cv2.cvtColor(pdata_, cv2.COLOR_BGR2HSV)
    # get mask
    hsv_mask  = cv2.inRange(hsv_pdata,lower,upper)
    # detect blue
    hsv_res  = cv2.bitwise_and(pdata_,pdata_,mask= hsv_mask)
    # plt BGR to RGB
    pdata_plt_hsv = cv2.cvtColor(hsv_res, cv2.COLOR_BGR2RGB)
    
    return pdata_plt_hsv

def WaveLet_ (img):
#     img_r = pywt.idwt2(img,"bior1.3")
    img_r = pywt.waverec2(img,"bior1.3")
    plt.imshow(img_r ,'gray')
    plt.show()
    cv2.imwrite('refactor.jpg',img_r)
    int_img_r = img_r.astype(int)
    return int_img_r

def CalcHist_(img):
    print(img.shape)
    img = img.astype('uint8')
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]) 
    plt.figure()
    plt.title('Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

 # Notice data type. If your data is float16 or float 32, it will cause not using for opencv function. 

def sobel_test(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0,ksize = 3)
    y = cv2.Sobel(img,cv2.CV_16S,0,1,ksize = 3 )
    absX = cv2.convertScaleAbs(x)   # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)  
    cv2.imwrite('./result/sobelX.jpg',absX)   
    cv2.imwrite('./result/sobelY.jpg',absY)   
    cv2.imwrite('./result/sobel.jpg',dst)
    return dst


def image_segmentation(img , kernel ,Original):
    
    Original_ = Original.copy()
    img_  =  img.copy()
    contours,__= cv2.findContours(img_,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for j in contours:
        area = cv2.contourArea(j)
        if area > 1:
#             print(j)
            cv2.drawContours(Original_,j, -1, (0, 255, 255), 2)
    
    cv2.imwrite('./result/result.jpg',Original_)
    plt.figure()
    plt.imshow(Original_)
    plt.savefig('./result/segmentation.jpg')
    plt.show()
    
    return img_ 
#endregion    

def Wavelet_SWT(img,filename):
    img1 = img.copy()
    level = pywt.swt_max_level(len(img))

    print('image level is {0}.'.format(level))
    titles = ['cA2','cH2','cV2','cD2','cA1','cH1','cV1','cD1']
    if level==0:
        return 0
    else:
        coeffs= pywt.swt2(img, 'bior3.3', level=level, start_level=0)   #wavelet  = 'haar' or any wavelet. bior3.3
        (cA2,(cH2,cV2,cD2)),(cA1,(cH1,cV1,cD1)) = coeffs
        coeffs_list = np.array([cA2,cH2,cV2,cD2,cA1,cH1,cV1,cD1])  #cA2 and cA1 is low frequency. [cA2,cH2,cV2,cD2,cA1,cH1,cV1,cD1]
        coeffs_np = coeffs_list.copy()


        #         # 将每个子图的像素范围都归一化到与CA2一致  CA2 [0,255* 2**level]
        # AH2 = np.concatenate([cA2, cH2+510], axis=1)
        # VD2 = np.concatenate([cV2+510, cD2+510], axis=1)
        # cA1 = np.concatenate([AH2, VD2], axis=0)

        # AH = np.concatenate([cA1, (cH1+255)*2], axis=1)
        # VD = np.concatenate([(cV1+255)*2, (cD1+255)*2], axis=1)
        # img = np.concatenate([AH, VD], axis=0)
        # plt.imshow(img,'gray')
        # plt.title('2D WT')
        # plt.show()

    # wave = pywt.Wavelet('bior3.3')
    # data_ = img[:]
    # (cA, cD) = pywt.dwt(data=data_, wavelet=wave)
    # print(cD)
    coeffs_1D= pywt.swt(img, 'bior3.3', level=level, start_level=0) 
    ((cA2,cD2),(cA1,cD1)) = coeffs_1D
    cD2_np = np.array(cD2)
    cD1_np = np.array(cD1)
    var1 = get_var(cD1_np)
    var2 = get_var(cD2_np)



    # print(var1)
    # print(var2)

    #show wavelet of image.
    # fig = plt.figure(figsize=(30, 30))
    # for i, a in enumerate([cA2,cH2,cV2,cD2,cA1,cH1,cV1,cD1]):
    #     ax = fig.add_subplot(4, 2, i + 1)
    #     ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)      
    #     ax.set_title(titles[i], fontsize=50)
    #     ax.set_xticks([])
    #     ax.set_yticks([])    
    # plt.savefig('result_multi_swt.jpg')
    # fig.tight_layout()
    # plt.show()

    print('cH2:{0}'.format(np.min(cH2)))
    print('cV2:{0}'.format(np.min(cV2)))
    print('cH1:{0}'.format(np.min(cH1)))
    print('cV1:{0}\n'.format(np.min(cV1)))


    for index,coeff in enumerate(coeffs_list):
        if index >0 and index <4:
            var = var2
        else:
            continue
        # thre = sure_shrink(var,coeff)
        # thre = visu_shrink(var2,coeff)
        thre = mini_max(var2,coeff)
        coeffs_np[index,:,:] = pywt.threshold(coeff,thre,'hard')

    # cD1[:,:] = 0
    # # cD2[:,:] = 0
    # # cV2[:,:] = 0
    # # cV1[:,:] = 0
    # threshold_para = 0.02
    # cH2 = pywt.threshold(cH2,threshold_para,'soft')
    # cV2 = pywt.threshold(cV2,threshold_para,'soft') 
    # cH1 = pywt.threshold(cH1,threshold_para,'soft') 
    # cV1 = pywt.threshold(cV1,threshold_para,'soft') 
    coeffs = (coeffs_np[0,:,:],(coeffs_np[1,:,:],coeffs_np[2,:,:],coeffs_np[3,:,:])),(coeffs_np[4,:,:],(coeffs_np[5,:,:],coeffs_np[6,:,:],coeffs_np[7,:,:]))

    # coeffs = (cA2,(cH2,cV2,cD2)),(cA1,(cH1,cV1,cD1))
    print('cH2:{0}'.format(np.min(coeffs_np[1,:,:])))
    print('cV2:{0}'.format(np.min(coeffs_np[2,:,:])))
    print('cH1:{0}'.format(np.min(coeffs_np[5,:,:])))
    print('cV1:{0}'.format(np.min(coeffs_np[6,:,:])))

    
    img_iswt = pywt.iswt2(coeffs,'bior3.3')
    img_iswt = img_iswt.astype(int)
    psnr_img = PSNR(img1, img_iswt, data_range=255)
    print('psnr:{0}'.format(psnr_img))
    cv2.imwrite('./result/swt_denoise.jpg',img_iswt)
    return img_iswt

# wavelet of denoise.
# region 
def get_var(cD):
    coeffs = np.abs(cD)
    np.sort(coeffs)
    row,col = coeffs.shape
    var = coeffs[int(np.ceil(row/2)),int(np.ceil(col/2))] / 0.6745
    return var
    
def sure_shrink(var, coeffs):
    N = coeffs.size
    sqr_coeffs = np.power(coeffs,2)
    sqr_coeffs  = (np.sort(sqr_coeffs)).flatten()
    pos = 0
    r = 0
    for idx, sqr_coeff in enumerate(sqr_coeffs):
        # for idx1, sqr_coeff_x in enumerate(sqr_coeff_y)
            new_r = (N - 2 * (idx + 1) + (N - (idx + 1))*sqr_coeff + sum(sqr_coeffs[0:idx+1])) / N
            if r == 0 or r > new_r:
                r = new_r
                pos = idx
    thre = math.sqrt(var) * math.sqrt(sqr_coeffs[pos])
    return thre

# 求VisuShrink法阈值
def visu_shrink(var, coeffs):
    # N = len(coeffs)
    N = coeffs.size
    thre = math.sqrt(var) * math.sqrt(2 * math.log(N))
    return thre

# 求HeurSure法阈值
def heur_sure(var, coeffs):
    # N = len(coeffs)
    # s = 0
    # for coeff in coeffs:
    #     s += math.pow(coeff, 2)
    N = coeffs.size
    s = np.sum(np.power(coeff2,2))
    theta = (s - N) / N
    miu = math.pow(math.log2(N), 3/2) / math.pow(N, 1/2)
    if theta < miu:
        return visu_shrink(var, coeffs)
    else:
        return min(visu_shrink(var, coeffs), sure_shrink(var, coeffs))

# 求Minimax法阈值
def mini_max(var, coeffs):
    # N = len(coeffs)
    N = coeffs.size
    if N > 32:
        return math.sqrt(var) * (0.3936 + 0.1829 * math.log2(N))
    else:
        return 0

# endregion

# edge detection.
#region

def contour2(Original,img,select):
    img_copy = img.copy()
    img_copy1 = img.copy()
    img_copy1 = img_copy1.astype('uint8')
    CalcHist_(img_copy)
    
    cv2.imwrite('./result/img_swt.jpg',img_copy)
    
    if(select == 'Gamma'):
        img_ = Gamma_transformation(img_copy,hsv_parmeter.gamma)
#         img_erode = cv2.erode(img_Gamma,hsv_parmeter.kernel,iterations = 10)
    elif(select == 'Linear'):
        img_ = linear_transformation(2,img_copy)    
#         img_erode = cv2.erode(img_linear,hsv_parmeter.kernel,iterations = 10)
#    
    cv2.imwrite('./result/img_.jpg',img_)
    img_erode = cv2.erode(img_,hsv_parmeter.kernel,iterations = 5)
    img_dilate = cv2.dilate(img_erode,hsv_parmeter.kernel,iterations = 5)
    cv2.imwrite('./result/img_dilate.jpg',img_dilate)
#     blurred = cv2.GaussianBlur(img_dilate, (5 ,5), 0)

    Laplacian_ =  cv2.Laplacian(img_,cv2.CV_16S,ksize = 5)
    cv2.imwrite('./result/Laplacian_.jpg',Laplacian_)
    sobel_ = sobel_test(img_)
    canny = cv2.Canny(img_, 80, 160)
    cv2.imwrite('./result/canny.jpg',canny)

    image_adaptiveThreshold = cv2.adaptiveThreshold(Laplacian_,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,1)
    inverse_image_adaptiveThreshold = 255-image_adaptiveThreshold
    inverse_image_adaptiveThreshold_copy = inverse_image_adaptiveThreshold.copy() 
    cv2.imwrite('./result/inverse_adaptiveThreshold.jpg',inverse_image_adaptiveThreshold_copy)
    contour_img = image_segmentation(canny,hsv_parmeter.kernel,Original)
#endregion

if __name__=='__main__':
    filename = 'test.jpg'
    pdata = cv2.imread(filename)
    pdata_copy = pdata.copy()

    red = hsv_transformation(pdata_copy , hsv_parmeter.lower,hsv_parmeter.upper)
    dot = hsv_transformation(pdata_copy , hsv_parmeter.lower1,hsv_parmeter.upper1)
    red_ = cv2.cvtColor(red , cv2.COLOR_HSV2BGR).astype(np.float32)
    dot_ = cv2.cvtColor(dot , cv2.COLOR_HSV2BGR).astype(np.float32)

    # cv2.imwrite('red.jpg',red_)
    # cv2.imwrite('dot.jpg',dot_)

    image_combine = red + dot

    pdata_gray = cv2.cvtColor(image_combine , cv2.COLOR_BGR2GRAY).astype(np.float32)
#     hist = cv2.calcHist([pdata_gray], [0], None, [256], [0, 256])  
    pdata_linear= linear_transformation(2 , pdata_gray)
    pdata_blur = cv2.GaussianBlur(pdata_linear,(5,5),0)

    eroded = cv2.erode(pdata_blur,hsv_parmeter.kernel,iterations = 1)    #erode 
    dilated = cv2.dilate(eroded,hsv_parmeter.kernel,iterations = 3)
    cv2.imwrite('./result/original.jpg',dilated)


    # sigma_est = estimate_sigma(dilated, channel_axis=-1, average_sigmas=True)
    # im_bayes = denoise_wavelet(dilated,wavelet='haar',method='BayesShrink', mode='soft',rescale_sigma=True)

    # psnr_noisy = peak_signal_noise_ratio(dilated, im_bayes)
    # print(im_bayes.shape)
    # cv2.imwrite('./result/im_bayes.jpg',im_bayes)
    coffes_iswt = Wavelet_SWT(dilated,filename)
    
    # contour2(pdata_copy,coffes_iswt,'Linear')   #you can select Gamma or Linear mode
    