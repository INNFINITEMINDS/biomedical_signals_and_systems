clear all; 
close all;

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 2_2
% by Zhiguo Zhang, 10/2013

%% ============================4.1=========================== %%
%% load and display signal
load assg2_anc
figure
grid on
N = length(signal);
%t = [1:N]/Fs;
subplot(211);
plot(1:N,signal)
xlabel('Time Index')
ylabel('Magnitude')
title('Signal');

subplot(212);
plot(1:N,reference)
xlabel('Time Index')
ylabel('Magnitude')
title('Reference');
%print('signal_reference_4_1','-dpng','-r300');

%% ============================4.2=========================== %%
%LMS
mu = 0.1; %step size
L = 16; %filter length
[c, y, e] = lms_filter(reference, signal, L, mu);
figure
plot(1:N, e);
xlabel('Time Index');ylabel('Magnitude');
title('Denoised Signal LMS');
%print('denoised_signal_lms_4_2','-dpng','-r300');

figure
hold on; box on;
plot(1:N,signal,'b');
plot(1:N,y,'g');
plot(1:N,e,'r');
legend('Signal','Output','Denoised Signal');
xlabel('Time Index'); ylabel('Magnitude');
title('Comparison LMS');
%print('denoised_signal_lms_compared_4_2','-dpng','-r300');

%% ============================4.3=========================== %%
%RLS
L = 16;
lambda = 0.99;
delta = 0.01;
[c,y,e] = rls_filter(reference, signal, L, lambda, delta);
figure
plot(1:N, e);
xlabel('Time Index');ylabel('Magnitude');
title('Denoised Signal RLS');
set(gca, 'ylim', [-0.65 0.4]);
%print('denoised_signal_rls_4_2','-dpng','-r300');

figure
hold on; box on;
plot(1:N,signal,'b');
plot(1:N,y,'g');
plot(1:N,e,'r');
legend('Signal','Output','Denoised Signal');
xlabel('Time Index'); ylabel('Magnitude');
title('Comparison RLS');
set(gca, 'ylim', [-0.65 0.4]);
%print('denoised_signal_rls_compared_4_2','-dpng','-r300');



