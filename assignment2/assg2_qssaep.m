clear all; 
close all;

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 2_1
% by Zhiguo Zhang, 10/2013

%% ============================1.1=========================== %%
%% load and display signal
load assg2_qssaep
figure
plot(t,data)
xlabel('Time (ms)')
ylabel('Amplitude (\muV)')
title('Signal')
%print('qssaep_signal_1_1','-dpng','-r300');

N = length(data);
nfft = 2^nextpow2(N);
%% perform STFT
winsize = 250; % window size
%nfft = 1024;   % # FFT points
[P, f] = stft(data, winsize, nfft, Fs);

%% display spectrogram
figure
imagesc(t,f,P)
colorbar
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
axis xy
grid on
set(gca,'ylim',[1 100]) % set the limits of frequency in the plot
title('Spectogram')
%print('qssaep_spectogram_1_1','-dpng','-r300');

%% ============================1.2=========================== %%
%% re-calculate STFT usiing in-house stft, window size = 50
winsize = 50; % window size
%nfft = 1024;   % # FFT points
[P, f] = stft(data, winsize, nfft, Fs);

%% display spectrogram, window size = 50
figure
subplot(2,2,1);
imagesc(t,f,P)
%colorbar
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
title('window size = 50');
axis xy
grid on
set(gca,'ylim',[0 150]) % set the limits of frequency in the plot
%print('qssaep_spectogram_1_2_win_50','-dpng','-r300');

%% re-calculate STFT usiing in-house stft, window size = 100
winsize = 100; % window size
%nfft = 1024;   % # FFT points
[P, f] = stft(data, winsize, nfft, Fs);

%% display spectrogram, window size = 100
subplot(2,2,2);
imagesc(t,f,P)
%colorbar
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
title('window size = 100');
axis xy
grid on
set(gca,'ylim',[0 150]) % set the limits of frequency in the plot
%print('qssaep_spectogram_1_2_win_100','-dpng','-r300');
%% re-calculate STFT usiing in-house stft, window size = 500
winsize = 500; % window size
%nfft = 1024;   % # FFT points
[P, f] = stft(data, winsize, nfft, Fs);

%% display spectrogram, window size = 500
subplot(2,2,3);
imagesc(t,f,P)
%colorbar
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
title('window size = 500');
axis xy
grid on
set(gca,'ylim',[0 150]) % set the limits of frequency in the plot
%print('qssaep_spectogram_1_2_win_500','-dpng','-r300');
%% re-calculate STFT usiing in-house stft, window size = 1000
winsize = 1000; % window size
%nfft = 1024;   % # FFT points
[P, f] = stft(data, winsize, nfft, Fs);

%% display spectrogram, window size = 1000
subplot(2,2,4);
imagesc(t,f,P)
%colorbar
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
title('window size = 1000');
axis xy
grid on
set(gca,'ylim',[0 150]) % set the limits of frequency in the plot
%print('qssaep_spectogram_1_2_win_1000','-dpng','-r300');
%print('qssaep_spectogram_1_2_win','-dpng','-r300');

%% ============================1.3=========================== %%
f = Fs/2 * linspace(0,1,nfft/2+1);
omega = 2;
sigma = 1;
[P_mwt] = mwt(data, f, Fs, omega, sigma); %CWT

%Display
figure
imagesc(t,f,P_mwt);
xlabel('Time (ms)')
ylabel('Frequency (Hz)')
title('Mortlet Wavelet');
axis xy
grid on
set(gca,'ylim',[0 150]) % set the limits of frequency in the plot
%print('qssaep_spectogram_1_3_mwt','-dpng','-r300');

