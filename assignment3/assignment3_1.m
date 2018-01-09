clear all; 
close all;
%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 3_1
% by Sheshan Aaron, 11/2014

%% ============================1.1=========================== %%
%% load and display signal
load assg3_emg

N = length(emg); %Number of samples
P = 4; % the number of features/variables
t = [1:N]/Fs;

%dimensions of emg is NxP
channel1 = emg(:,1);
channel2 = emg(:,2);
channel3 = emg(:,3);
channel4 = emg(:,4);

figure
plot(t, channel1)
title('Channel 1') 
xlabel('Time (s)')
ylabel('Amplitude (mV)')
set(gca, 'LineWidth', 1)
grid on
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
print('1_1_emg_channel1','-dpng','-r300');

figure
plot(t, channel2)
title('Channel 2') 
xlabel('Time (s)')
ylabel('Amplitude (mV)')
set(gca, 'LineWidth', 1)
grid on
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
print('1_1_emg_channel2','-dpng','-r300');

figure
plot(t, channel3)
title('Channel 3') 
xlabel('Time (s)')
ylabel('Amplitude (mV)')
set(gca, 'LineWidth', 1)
grid on
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
print('1_1_emg_channel3','-dpng','-r300');

figure
plot(t, channel4)
title('Channel 4') 
xlabel('Time (s)')
ylabel('Amplitude (mV)')
set(gca, 'LineWidth', 1)
grid on
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
print('1_1_emg_channel4','-dpng','-r300');

%% ============================1.2=========================== %%
% Compute covariance, correlation, and partial correlation
C_cov = cov(emg); % covariance
C_corr = corr(emg); % correlation (Pearson's)
C_partialcorr = partialcorr(emg); % partial correlation (Pearson's)
%% ============================1.3=========================== %%
%% Compute PSDs (Welch’s method) and coherence
fs = Fs; % sampling rate
nfft = 2^nextpow2(N); % number of FFT points

%convert from mV to uV
channel1 = channel1*1000;
channel2 = channel2*1000;
channel3 = channel3*1000;
channel4 = channel4*1000;
% calculate PSD using Welch's Method
windowLen = 100;%set window length 1to 100
%default nooverlap is 50%
[P1, f1] = pwelch(channel1,windowLen,[],nfft,fs); % Welch's method
[P2, f2] = pwelch(channel2,windowLen,[],nfft,fs); % Welch's method
[P3, f3] = pwelch(channel3,windowLen,[],nfft,fs); % Welch's method
[P4, f4] = pwelch(channel4,windowLen,[],nfft,fs); % Welch's method

%plot PSDs
figure
subplot(2,2,1);
plot(f1, P1);set(gca,'xlim',[0 230])
xlabel('Frequency (Hz)')
ylabel('PSD (\muV^2/Hz)')
title('PSD Channel 1')
grid on

subplot(2,2,2);
plot(f2, P2);set(gca,'xlim',[0 230])
xlabel('Frequency (Hz)')
ylabel('PSD (\muV^2/Hz)')
title('PSD Channel 2')
grid on

subplot(2,2,3);
plot(f3, P3);set(gca,'xlim',[0 230])
xlabel('Frequency (Hz)')
ylabel('PSD (\muV^2/Hz)')
title('PSD Channel 3')
grid on

subplot(2,2,4);
plot(f4, P4);set(gca,'xlim',[0 230])
xlabel('Frequency (Hz)')
ylabel('PSD (\muV^2/Hz)')
title('PSD Channel 4')
grid on

set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
print('1_3_PSDs','-dpng','-r300');

%MSC
figure
COH = zeros(4,4);
f = zeros(4,4);
count = 1;
for i = 1 : 4;
    for j = 1 : 4;
        [COH, f] =  mscohere(emg(:,i),emg(:,j),windowLen,[],nfft,fs); % Magnitude
        subplot(4,4,count);
        plot(f, COH);set(gca,'xlim',[0 250])
        if(j == 1)
            ylabel(sprintf('Channel %i \n MSC', i), 'FontSize', 12);
        end
        if(i == 1)
            title(sprintf('Channel %i', j), 'FontSize', 12);
        end
        if(i == 4)
            xlabel('Frequency (Hz)', 'FontSize', 12);
        end
        grid on        
        count = count + 1;
    end
end

set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [5 5]); %Set the paper to have width 5 and height 5.
%print('1_3_MSCs','-dpng','-r300');

% [COH1x1,f1x1] = mscohere(channel1,channel1,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH1x2,f1x2] = mscohere(channel1,channel2,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH1x3,f1x3] = mscohere(channel1,channel3,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH1x4,f1x4] = mscohere(channel1,channel4,windowLen,[],nfft,fs); % Magnitude squared coherence
% 
% [COH2x1,f2x1] = mscohere(channel2,channel1,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH2x2,f2x2] = mscohere(channel2,channel2,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH2x3,f2x3] = mscohere(channel2,channel3,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH2x4,f2x4] = mscohere(channel2,channel4,windowLen,[],nfft,fs); % Magnitude squared coherence
% 
% [COH3x1,f3x1] = mscohere(channel3,channel1,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH3x2,f3x2] = mscohere(channel3,channel2,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH3x3,f3x3] = mscohere(channel3,channel3,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH3x4,f3x4] = mscohere(channel3,channel4,windowLen,[],nfft,fs); % Magnitude squared coherence
% 
% [COH4x1,f4x1] = mscohere(channel4,channel1,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH4x2,f4x2] = mscohere(channel4,channel2,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH4x3,f4x3] = mscohere(channel4,channel3,windowLen,[],nfft,fs); % Magnitude squared coherence
% [COH4x4,f4x4] = mscohere(channel4,channel4,windowLen,[],nfft,fs); % Magnitude squared coherence
% 
% %plot MSCs
% %channel 1
% figure
% subplot(4,4,1);
% plot(f1x1, COH1x1)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch1 x Ch1)')
% grid on
% 
% subplot(4,4,2);
% plot(f1x2, COH1x2)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch1 x Ch2)')
% grid on
% 
% subplot(4,4,3);
% plot(f1x3, COH1x3)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch1 x Ch3)')
% grid on
% 
% subplot(4,4,4);
% plot(f1x4, COH1x4)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch1 x Ch4)')
% grid on
% 
% %channel 2
% subplot(4,4,5);
% plot(f2x1, COH2x1)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch2 x Ch1)')
% grid on
% 
% subplot(4,4,6);
% plot(f2x2, COH2x2)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch2 x Ch2)')
% grid on
% 
% subplot(4,4,7);
% plot(f2x3, COH2x3)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch2 x Ch3)')
% grid on
% 
% subplot(4,4,8);
% plot(f2x4, COH2x4)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch2 x Ch4)')
% grid on
% 
% %channel 3
% subplot(4,4,9);
% plot(f3x1, COH3x1)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch3 x Ch1)')
% grid on
% 
% subplot(4,4,10);
% plot(f3x2, COH3x2)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch3 x Ch2)')
% grid on
% 
% subplot(4,4,11);
% plot(f3x3, COH3x3)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch3 x Ch3)')
% grid on
% 
% subplot(4,4,12);
% plot(f3x4, COH3x4)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch3 x Ch4)')
% grid on
% 
% %channel 4
% subplot(4,4,13);
% plot(f4x1, COH4x1)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch4 x Ch1)')
% grid on
% 
% subplot(4,4,14);
% plot(f4x2, COH4x2)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch4 x Ch2)')
% grid on
% 
% subplot(4,4,15);
% plot(f4x3, COH4x3)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch4 x Ch3)')
% grid on
% 
% subplot(4,4,16);
% plot(f4x4, COH4x4)
% xlabel('Frequency (Hz)')
% ylabel('MSC')
% title('Coherence (Ch4 x Ch4)')
% grid on