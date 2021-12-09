function ADSP_HW4_M10907305()
    L = 255;
    A = input('input image of A:');
    B = input('input image of B:');
    c1 = input('input c1:');
    c2 = input('input c2:');
    ssim = SSIM(A,B,c1,c2)
end

function ssim = SSIM(A,B,c1,c2)
    L = 255;
    [M,N] =size(A);                        %define image of size to [M,N].
    img1_gray = double(rgb2gray(A));       %define image of A convert gray.
    img2_gray = double(rgb2gray(B));       %define image of B convert gray.
    means_img1 = mean(mean(img1_gray));       %define mean of x.
    means_img2 = mean(mean(img2_gray));       %define mean of y.
    variances_img1 = (sum(sum((img1_gray-means_img1).^2)))/(M*N);   %define variance of x.
    variances_img2 = (sum(sum((img2_gray-means_img2).^2)))/(M*N);   %define variance of y.
    convariance_img12 = sum(sum( (img1_gray-means_img1).*(img2_gray-means_img2) ))/(M*N); %define covariance of x and y.
    ssim = (2*means_img1*means_img2 + (c1*L)^2)*(2*convariance_img12+(c2*L)^2)/(variances_img1+variances_img2+(c2*L)^2)/(means_img1^2+means_img2^2+(c1*L)^2);

    subplot(1,2,1),imshow(rgb2gray(A)),title('Original image');
    subplot(1,2,2),imshow(rgb2gray(B)),title({'Comparison image';'SSIM';ssim});
end


