clear all; 
close all;

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 2_2
% by Zhiguo Zhang, 10/2013

%% ============================2.1=========================== %%
%% load and display signal
load assg2_ecg
figure
N = length(signal);
t = [1:N]/Fs;
plot(t,signal)
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
grid on
%print('ecg_signal_2_1','-dpng','-r300');

%% ============================2.2=========================== %%
%Level 5 decompostion, using db5 wavelet
[C,L] = wavedec(signal,5,'db5');

%Reconstruct level 5 approximation and Level 1-5 details

%Reconstruct Level 5 approximation
A5 = wrcoef('a',C,L,'db5',5);

%Reconstruct Level 1-5 details
D1 = wrcoef('d',C,L,'db5',1); 
D2 = wrcoef('d',C,L,'db5',2); 
D3 = wrcoef('d',C,L,'db5',3);
D4 = wrcoef('d',C,L,'db5',4);
D5 = wrcoef('d',C,L,'db5',5);

%Diplay approximations and decomposition
figure
subplot(2,3,1); plot(t,A5); 
title('Approximation A5') 
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
subplot(2,3,2); plot(t,D1); 
title('Detail D1') 
xlabel('Time (s)')
ylabel('Magnitude')
subplot(2,3,3); plot(t,D2); 
title('Detail D2')
xlabel('Time (s)')
ylabel('Magnitude')
subplot(2,3,4); plot(t,D3); 
title('Detail D3')
xlabel('Time (s)')
ylabel('Magnitude')
subplot(2,3,5); plot(t,D4); 
title('Detail D4')
xlabel('Time (s)')
ylabel('Magnitude')
subplot(2,3,6); plot(t,D5); 
title('Detail D5')
xlabel('Time (s)')
ylabel('Magnitude')
%print('level5_wavelet_decomposition_2_2','-dpng','-r300');

%% ============================2.3=========================== %%
%wavelet thresholding, denoising
[thr,sorh,keepapp] = ddencmp('den','wv',signal); %calc default denoising parameters
clean = wdencmp('gbl',C,L,'db5',5,thr,sorh,keepapp); %perform actual denoising

%display
figure
subplot(2,1,1); plot(t, signal); title('Original')
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
subplot(2,1,2); plot(t, clean); title('Denoised')
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
%print('denoised_vs_original_2_3','-dpng','-r300');

