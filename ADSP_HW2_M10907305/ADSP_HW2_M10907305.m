function ADSP_HW2_M10907305(k,f_sampling,transition_band)
%example:ADSP_HW2_M10907305(10,1000,0.1)
%transition_band is Symmetrical,so you cann't set two 
% How to use a k point to design.
% f_sampling is sampling of freguency.
N = 2*k+1;
F = [ (0:k)*(1-transition_band)/N (k+1:N-1)*(1-transition_band)/N+transition_band];

Hd_f = j*2*pi*(F-(F>=0.5));
r1 = ifft(Hd_f);
r  = circshift(r1,k);

Hd_sampling1 = (0:f_sampling*(1-transition_band)/2)/f_sampling;
Hd_sampling2 = (f_sampling*(transition_band+(1-transition_band)/2):f_sampling)/f_sampling;
Hd_f_sampling1 = j*2*pi*(Hd_sampling1-(Hd_sampling1>=0.5));
Hd_f_sampling2 = j*2*pi*(Hd_sampling2-(Hd_sampling2>=0.5));
transition = linspace(Hd_f_sampling1(end),Hd_f_sampling2(1),transition_band*f_sampling+1);
Hd_f_sampling = [ Hd_f_sampling1(1:end-1) transition  Hd_f_sampling2(1:end-1)];
ff = [];
for i = 1:N
    ff = [ff Hd_f_sampling(round((i-1)/N*1000)+1) ]  ;
end
fff = circshift(ifft(ff),k);
fftt = fft(circshift([fff zeros(1, f_sampling-2*k-1)], [0 -k]));
fs_sampling = (0:f_sampling-1)/f_sampling;

figure;
% plot([F 1],imag([Hd_f 0]),'blue');
plot((0:f_sampling)/f_sampling,imag(Hd_f_sampling),'blue',fs_sampling,imag(fftt),'red');
title('Frequency Response')
xlabel('F')
figure;
subplot(3,1,1);
stem(0:N-1,real(r1));
xlim([-N-5,N+5]);
title('r1[n]');
subplot(3,1,2);
stem(-k:k,real(r));
xlim([-N-5,N+5]);
title('r[n]');
subplot(3,1,3);
stem(0:N-1,real(r));
xlim([-N-5,N+5]);
title('h[n]');


