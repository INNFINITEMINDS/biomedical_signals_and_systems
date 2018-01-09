clear all;
close all;

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 1
% by Sheshan Aaron, 10/2014

%% Load data and specify parameters
load assg1_eegdata % load data
N = length(eeg); % data length
t = [1:N]/Fs; % time axis
%% Display EMG data
figure
plot(t,eeg)
xlabel('Time (sec)')
ylabel('Amplitude (mV)')
title('EEG Waveform')
set(gca,'xlim',[0, max(t)]) % set the limits of time in the plot
%% Calculate periodogram of original signal
x = detrend(eeg); % detrending
nfft = 2^nextpow2(N); % number of FFT points
[P_per, f] = periodogram(x,[],nfft,Fs); % calculate periodogram
%% Calculate PSD of original signal using Welch Method
nfft = 2^nextpow2(N);
[wPS,wF] = pwelch(detrend(eeg),hamming(500),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
%% Calculate AR-based PSD of original signal using Yule-Walker Method
%model order selection AIC
for p = 1:200 %model order from 1 to 200
    [junk_ar_coeffs, NoiseVariance(p)] = aryule(eeg, p);
    aic(p) = N*log(NoiseVariance(p)) + 2*p;
    %bic(p) = N*log(NoiseVariance(p)) + log(N)*p;
    %fpe(p) = (NoiseVariance(p))*(N+p+1)/(N-p-1);
end
[junk, p_aic] = min(aic); %optimal order seleted by AIC
%[junk, p_bic] = min(aic); %optimal order seleted by BIC
%[junk, p_fpe] = min(aic); %optimal order seleted by FPE

p = p_aic; %model order
nfft =  2^nextpow2(N);
[arPS, arF] = pyulear(eeg, p, nfft, Fs);
%% PSD of orginal signal using Periodogram, Welch and AR (Yule-Walker) Methods
figure
hold on
plot(f,10*log10(P_per),'g')
plot(arF,10*log10(arPS),'b')
plot(wF,10*log10(wPS),'r')
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Welch Method and AR Method')
set(gca,'xlim',[0, 100],'ylim',[-35 25]) % set the limits of frequency and PSD in the plot
legend('PSD using periodogram','PSD using AR method','PSD using Welch Method')
%% highpass filter
Wp = 3/(Fs/2);
Ws = 1/(Fs/2);
Rp = 3;
Rs = 60;
[n, Wn] = buttord(Wp,Ws,Rp, Rs);
%[b a] = ellip(n, Rp, Rs, Wn, 'high');
[b a] = butter(n, Wn, 'high');
[h,f] = freqz(b, a, N, Fs);
h = 20*log10(abs(h));
figure
semilogx(f,h, 'b'); axis tight
plot(f, h)
xlabel('Frequency (Hz)'); ylabel('X(f)(dB)');
title('Butterworth Highpass filter');

%apply filter to original signal
y = filtfilt(b, a, eeg);
subplot(2,2,1);
plot(t,y)
xlabel('Time (sec)')
ylabel('Amplitude (mV)')
title('EEG Waveform with Highpass Butterworth Filter')
set(gca,'ylim',[-40, 40]) % set the limits of time in the plot

% Calculate PSD of highpass signal using periodogram
x = detrend(y); % detrending
nfft = 2^nextpow2(N); % number of FFT points
[P_per, f] = periodogram(x,[],nfft,Fs); % calculate periodogram
% Display PSD of filtifilt signal using periodogram
subplot(2,2,2);
plot(f,10*log10(P_per))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Highpass Filter Periodogram')
set(gca,'xlim',[0, Fs/2],'ylim',[-70 20]) % set the limits of frequency and PSD in the plot

% Calculate PSD of highpass signal using Welch Method
nfft = 2^nextpow2(N);
[PS,F] = pwelch(detrend(y),hamming(500),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
subplot(2,2,3);
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Highpass Filter Welch Method')
set(gca,'xlim',[0, Fs/2],'ylim',[-70 20]) % set the limits of frequency and PSD in the plot

%Calculate AR-based PSD of highpass signal using Yule-Walker Method
%model order selection AIC
for p = 1:200 %model order from 1 to 200
    [junk_ar_coeffs, NoiseVariance(p)] = aryule(y, p);
    aic(p) = N*log(NoiseVariance(p)) + 2*p;
    %bic(p) = N*log(NoiseVariance(p)) + log(N)*p;
    %fpe(p) = (NoiseVariance(p))*(N+p+1)/(N-p-1);
end
[junk, p_aic] = min(aic); %optimal order seleted by AIC
%[junk, p_bic] = min(aic); %optimal order seleted by BIC
%[junk, p_fpe] = min(aic); %optimal order seleted by FPE

p = p_aic; %model order
nfft =  2^nextpow2(N);
[PS, F] = pyulear(y, p, nfft, Fs);
subplot(2,2,4);
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Highpass Filter AR Model (Yule-Walker Method)')
set(gca,'xlim',[0, Fs/2],'ylim',[-70 20]) % set the limits of frequency and PSD in the plot

%% lowpass filter
Wp = 30/(Fs/2);
Ws = 50/(Fs/2);
Rp = 3;
Rs = 60;
[n, Wn] = buttord(Wp,Ws,Rp, Rs);
%[c d] = ellip(n, Rp, Rs, Wn, 'low');
[c d] = butter(n, Wn, 'low');
[h,f] = freqz(c, d, N, Fs);
h = 20*log10(abs(h));
figure
semilogx(f,h, 'b'); axis tight
plot(f, h)
xlabel('Frequency (Hz)'); ylabel('X(f)(dB)');
title('Butterworth Lowpass filter');
%apply filter to original signal
y2 = filtfilt(c, d, y);
subplot(2,2,1);
plot(t,y2)
xlabel('Time (sec)')
ylabel('Amplitude (mV)')
title('EEG Waveform with Lowpass Butterworth Filter')
set(gca,'ylim',[-40, 50]) % set the limits of time in the plot

% Calculate PSD of lowpass signal using periodogram
x = detrend(y2); % detrending
nfft = 2^nextpow2(N); % number of FFT points
[P_per, f] = periodogram(x,[],nfft,Fs); % calculate periodogram
% Display PSD of filtifilt signal using periodogram
subplot(2,2,2);
plot(f,10*log10(P_per))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Lowpass Filter Periodogram')
set(gca,'xlim',[0, Fs/2],'ylim',[-70 30]) % set the limits of frequency and PSD in the plot

% Calculate PSD of lowpass signal using Welch Method
nfft = 2^nextpow2(N);
[PS,F] = pwelch(detrend(y2),hamming(500),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
subplot(2,2,3);
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Lowpass Filter Welch Method')
set(gca,'xlim',[0, Fs/2],'ylim',[-70 30]) % set the limits of frequency and PSD in the plot

%Calculate AR-based PSD of lowpass signal using Yule-Walker Method
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
ssssnfft =  2^nextpow2(N);
[PS, F] = pyulear(y2, p, nfft, Fs);
subplot(2,2,4);
plot(F,10*log10(PS))
xlabel('Frequency (Hz)')
ylabel('PSD (dB)')
title('Lowpass Filter AR Model (Yule-Walker Method)')
set(gca,'xlim',[0, Fs/2],'ylim',[-70 30]) % set the limits of frequency and PSD in the plot
