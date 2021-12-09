function ADSP_HW5_M10907305()
x = input('input the real signal of f1(row vector):');     % ex: f1 = [10:50]
y = input('input the real signal of f2(row vector):');     % ex: f2 = [50:90]
[Fx,Fy] = fftreal(x,y)
function [Fx,Fy] = fftreal(x,y)
    while (length(x) ~= length(y)) 
        if (length(x) ~= length(y) )
            fprintf('\nWrong length,please input your the real signals again.\n')
            x = input('input the real signal of f1(row vector):');
            y = input('input the real signal of f2(row vector):');
        end
    end   
    f3 = x + 1i*y;
    FFT_f3 = fft(f3);
    Fx = (FFT_f3 + conj(circshift(fliplr(FFT_f3),[0 1])))/2;        % even sequence
    Fy = (FFT_f3 - conj(circshift(fliplr(FFT_f3),[0 1])))/(2*1i);   % odd sequence

