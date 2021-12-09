function ADSP_HW3_M10907305()
%please enter your image location.
f = imread('Lenna.jpg');
% A = [0.299 0.857 0.114 ;-0.169 -0.331 0.5 ; 0.5 -0.419 -0.081];
% Y_ = f(:,:,1)*A(1,1) + f(:,:,2)*A(1,2) + f(:,:,3)*A(1,3);
% Cb_ = f(:,:,1)*A(2,1)+ f(:,:,2)*A(2,2) + f(:,:,3)*A(2,3) ;
% Cr_ = f(:,:,1)*A(3,1) + f(:,:,2)*A(3,2) + f(:,:,3)*A(3,3);
% Digtal RGB to YCbCr
Y_ = 16+(65.481/255*f(:,:,1)) + (128.553/255*f(:,:,2)) + (24.966/255*f(:,:,3));
Cb_ = 128-(37.797/255*f(:,:,1)) - (74.203/255*f(:,:,2)) + (112.0/255*f(:,:,3));
Cr_ = 128+(112.0/255*f(:,:,1)) - (93.786/255*f(:,:,2)) - (18.214/255*f(:,:,3));
img_ycbcr =  cat(3,Y_,Cb_,Cr_);
[row,col,~] = size(img_ycbcr);
Y(:,:,1) = img_ycbcr(:,:,1);
% 4:2:0
for i=1:2:row-1
    for j=1:2:col-1 
        Cb((i+1)/2,(j+1)/2)=double(img_ycbcr(i,j,2)); 
    end
end
for i=2:2:row
    for j=1:2:col-1 
        Cr(i/2,(j+1)/2)=double(img_ycbcr(i,j,3));   
    end
end

% reconstructed image
for i=1:row/2
    for j=1:col/2
        Y(2*i-1,2*j-1,2)=Cb(i,j);
        Y(2*i,2*j-1,2)=Cb(i,j);
        Y(2*i-1,2*j,2)=Cb(i,j);
        Y(2*i,2*j,2)=Cb(i,j);
        
        Y(2*i-1,2*j-1,3)=Cr(i,j);
        Y(2*i,2*j-1,3)=Cr(i,j);
        Y(2*i-1,2*j,3)=Cr(i,j);
        Y(2*i,2*j,3)=Cr(i,j); 
    end
end
% Y = 16+(65.481/255*f(:,:,1)) + (128.553/255*f(:,:,2)) + (24.966/255*f(:,:,3));
% Cb = 128-(37.797/255*f(:,:,1)) - (74.203/255*f(:,:,2)) + (112.0/255*f(:,:,3));
% Cr = 128+(112.0/255*f(:,:,1)) - (93.786/255*f(:,:,2)) - (18.214/255*f(:,:,3));
% img_rec =  cat(3,Y,Cb,Cr);

% YCbCr to RGB
img_rec = double(Y);
R = 1.164*(img_rec(:,:,1)-16) + 1.596*(img_rec(:,:,3)-128);
G = 1.164*(img_rec(:,:,1)-16) - 0.813*(img_rec(:,:,3)-128) - 0.392*(img_rec(:,:,2)-128);
B = 1.164*(img_rec(:,:,1)-16) + 2.017*(img_rec(:,:,2)-128);
% R = img_rec(:,:,1) + 1.402*img_rec(:,:,3);
% G = img_rec(:,:,1) - 0.344*img_rec(:,:,2) - 0.714*img_rec(:,:,3);
% B = img_rec(:,:,1) + 1.772*img_rec(:,:,2);
img_recc =  cat(3,R,G,B);
img_recc = uint8(img_recc);

MAX=255;
MES=sum(sum(sum((f-img_recc).^2)))/(row*col*3);    %均方差
PSNR=20*log10(MAX/sqrt(MES))             %峰值信噪比

psnr(img_recc,f)

subplot(1,2,1),imshow(f),title('original image');
subplot(1,2,2),imshow(img_recc),title({'reconstructed image';'RGB:PSNR';num2str(PSNR)});


                                                                                                        