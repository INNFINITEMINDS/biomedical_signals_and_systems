clear all;
close all;

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 1
% by Zhiguo Zhang, 09/2013

%% Load data and specify parameters
load assg1_emgdata % load data
N = length(emg); % data length
t = [1:N]/Fs; % time axis
%% Display EMG data
figure
plot(t,emg)
xlabel('Time (sec)')
ylabel('Amplitude (mV)')
title('EMG Waveform')
set(gca,'xlim',[0, max(t)]) % set the limits of time in the plot
%% Calculate PSD of original signal using Welch Method
nfft = 2^nextpow2(N);
[PS,F] = pwelch(detrend(emg),hamming(5000),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
figure
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Welch Method')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
%% Calculate AR-based PSD of original signal using Yule-Walker Method
%model order selection AIC
for p = 1:200 %model order from 1 to 200
    [junk_ar_coeffs, NoiseVariance(p)] = aryule(emg, p);
    aic(p) = N*log(NoiseVariance(p)) + 2*p;
end
[junk, p_aic] = min(aic); %optimal order seleted by AIC
p = 500; %model order
nfft =  2^nextpow2(N);
[PS, F] = pyulear(emg, p, nfft, Fs);
figure
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('AR Model (Yule-Walker Method)')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
%% 6-order Butterworth bandpass filter
wn = [10 250]/(Fs/2);
[b, a] = butter(3,wn,'bandpass');
[h,f] = freqz(b, a, N, Fs);
h = 20*log10(abs(h));
figure
semilogx(f,h, 'b'); axis tight
plot(f, h)
xlabel('Frequency (Hz)'); ylabel('X(f)(dB)');
title('Butterworth');

%apply filter to original signal
y = filter(b, a, emg);
y2 = filtfilt(b, a, emg);

N = length(emg); % data length
t = [1:N]/Fs; % time axis %select 101 to 200
figure
hold on
plot(t,emg,'r')
plot(t,y,'g')
plot(t,y2,'b')
xlabel('Time (sec)')
ylabel('Amplitude (mV)')
title('EMG Waveform with Bandpass Butterworth Filter')
set(gca,'xlim',[101/Fs, 200/Fs]) % set the limits of time in the plot
legend('emg signal','filter', 'filtfilt')

%% Calculate PSD of filtfilt signal using periodogram
x = detrend(y2); % detrending
nfft = 2.^nextpow2(N); % number of FFT points
[P_per, f] = periodogram(x,[],nfft,Fs); % calculate periodogram

%% Display PSD of filtifilt signal using periodogram
figure
plot(f,10*log10(P_per))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Signal through filtfilt Periodogram')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
%% Calculate PSD of filtfilt signal using Welch Method
nfft = 2^nextpow2(N);
[PS,F] = pwelch(detrend(y2),hamming(5000),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
figure
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Signal through filtfilt Welch Method')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
%% Calculate AR-based PSD of filtfilt signal using Yule-Walker Method
%model order selection AIC
for p = 1:200 %model order from 1 to 200
    [junk_ar_coeffs, NoiseVariance(p)] = aryule(y2, p);
    aic(p) = N*log(NoiseVariance(p)) + 2*p;
    %bic(p) = N*log(NoiseVariance(p)) + log(N)*p;
    %fpe(p) = (NoiseVariance(p))*(N+p+1)/(N-p-1);
end
[junk, p_aic] = min(aic); %optimal order seleted by AIC
%[junk, p_bic] = min(aic); %optimal order seleted by BIC
%[junk, p_fpe] = min(aic); %optimal order seleted by FPE

p = p_aic; %model order
nfft =  2^nextpow2(N);
[PS, F] = pyulear(y2, p, nfft, Fs);
figure
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Signal through filtfilt AR Model (Yule-Walker Method)')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
%% PSD of original signal vs filtfilt signal using Welch Method
nfft = 2^nextpow2(N);
[PS0,F0] = pwelch(detrend(emg),hamming(5000),[],nfft, Fs);%default is hamming window with 50% overlap, original signal
[PS1,F1] = pwelch(detrend(y2),hamming(5000),[],nfft, Fs);%default is hamming window with 50% overlap, filtered signal

figure
hold on
plot(F0,10*log10(PS0), 'b')
plot(F1,10*log10(PS1),'r')
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('PSD or original signal vs filtered signal Welch Method')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
legend('original emg signal', 'filtfilt filtered emg signal')
%% Calculate periodogram of original signal
x = detrend(emg); % detrending
nfft = 2.^nextpow2(N); % number of FFT points
[P_per, f] = periodogram(x,[],nfft,Fs); % calculate periodogram
%% Display periodogram of original signal
figure
plot(f,10*log10(P_per))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('EMG Periodogram')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
%% Downsample EMG Signal
emg = decimate(emg,4); %orignal Fs is 2000, reduce by 4 to Fs of 500
Fs = 500;
N = length(emg); % data length
t = [1:N]/Fs; % time axis
figure
plot(t,emg)
xlabel('Time (sec)')
ylabel('Amplitude (mV)')
title('Downsampled EMG Waveform')
set(gca,'xlim',[0, max(t)]) % set the limits of time in the plot
%% Calculate periodogram of downsampled signal
x = detrend(emg); % detrending
nfft = 2.^nextpow2(N); % number of FFT points
[P_per, f] = periodogram(x,[],nfft,Fs); % calculate periodogram

%% Display periodogram of downsampled signal
figure
plot(f,10*log10(P_per))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Downsampled EMG Periodogram')
set(gca,'xlim',[0, Fs/2],'ylim',[-100 -20]) % set the limits of frequency and PSD in the plot
