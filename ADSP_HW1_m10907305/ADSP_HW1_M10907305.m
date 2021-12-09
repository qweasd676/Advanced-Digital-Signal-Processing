function ADSP_HW1_M10907305(N,delta,F)
% ADSP_HW1_M10907305(19,0.0001,[0 0.05 0.09 0.13 0.175 0.25 0.26 0.33 0.38 0.43 0.5 ])
clc
K = (N-1) / 2;   % 
transition_band_upper = 0.225;  %0.225
transition_band_lower = 0.175;  %0.175
transition_band_center = 0.2;   %0.2
W_lower =0.5;  %0.5
W_upper = 1;   %1
f_sampling = 0:delta:0.5;  %delta
H_sampling = f_sampling > transition_band_center ;        %setting

E1 = 9999;
E0 = 99;
aa = 0;
period = [0:K];
item_k = [0:K+1];
E0_register = [];

while( (0 > E1 - E0) || (E1 - E0 > delta ) )
    if( aa == 1)
        E1 = E0;
    end
    W = W_lower*(F<=transition_band_lower & F>= 0) + W_upper*(F>=transition_band_upper & F<= 0.5); %Setting range of F
    H = 1*(F>=transition_band_upper & F<= 0.5);          %Setting range of F
    %H = 1*(F<=transition_band_lower & F>= 0)          %Setting range of F
    s= inv(([cos(2*pi*F.'*period)  [(-1).^item_k./W].'])) *H.';
    RF_sampling = 0;
    RF = 0;
    for i = 1:K+1
        RF_sampling = RF_sampling + s(i)*cos(2*pi*(i-1)*f_sampling);
        RF = RF + s(i)*cos(2*pi*(i-1)*F);
    end
    %RF
    W_sampling = W_lower*(f_sampling<=transition_band_lower & f_sampling>= 0) + W_upper*(f_sampling>=transition_band_upper & f_sampling<= 0.5);
    err_sampling = [RF_sampling - H_sampling ].*W_sampling;
    err_sampling_zero = [0 err_sampling 0];
    err = [RF - H ].*W;
    P = [];
    for i = 2:length(err_sampling_zero)-1
        if( (err_sampling_zero(i) > err_sampling_zero(i+1)) & (err_sampling_zero(i) > err_sampling_zero(i-1)) )
            P = [P f_sampling(i-1)];
        elseif( (err_sampling_zero(i) < err_sampling_zero(i+1)) & (err_sampling_zero(i) < err_sampling_zero(i-1)) )
            P = [P f_sampling(i-1)];
        end
    end
    %P
    E0 = max(abs(err_sampling));
    E0_register = [E0_register E0];
    P1 = [];
    P2 = [];
    if((E1 - E0 > delta) || (E1 - E0 < 0))
        if(length(P) > K+2)
             for i = 1:length(P)
                if( (P(i) ~= 0) &  (P(i) ~= 0.5) & (P(i) ~= transition_band_lower) & (P(i) ~= transition_band_upper) )
                    P1 = [P1 P(i)];
                else 
                    P2 = [P2 P(i)];
                end            
             end
             %P1
             %P2
             if(length(P1) < K+2)
                 point = (K+2)  -length(P1) ;   
                 P_abs_err = abs((err_sampling(P2/delta +1)));
                 [b ,c ] = sort(P_abs_err,'descend');
                 P  = sort([P1 P2(c(1:point))]);
             end            
        end
        F = P;
        aa = 1;
    end
    %P
end
E0_register
subplot(2,1,1)
h_f = [(fliplr(s(2:end-1).'))/2 s(1) s(2:end-1).'/2];
x = 0:1:length(h_f)-1;
stem(x,h_f)
xlim([-1 length(h_f)+1])
ylim([min(h_f)-0.1 max(h_f)+0.1])
title('Impulse Response')
xlabel('time')

subplot(2,1,2)
plot(f_sampling,H_sampling,'k',f_sampling,RF_sampling,'b')
axis([0,0.5,-0.5,1.5])
title('Frequency Response')
xlabel('Frequency(Hz)')
for i = 2:length(F)-1
    xline(F(i),'--');
end








