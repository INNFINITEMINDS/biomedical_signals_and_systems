clear all; 
close all;
%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 3_4
% by Sheshan Aaron, 11/2014

%% ============================4.1=========================== %%
%%Load the data
load assg3_eeg_classification

%% Pre-processing Bandpass filter to remove noise and artifacts
Fs = 1000; %new sample rate is 1000

N = length(eeg_ec); %since N_ec has same value as N_eo
t = [1:N]/Fs; %since t_ec has same value as t_eo

figure
plot(t, (eeg_ec));
title('Raw EEG EC') 
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
set(gca, 'LineWidth', 1)
grid on
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
print('4_1_raw_eeg_ecc','-dpng','-r300');

figure
plot(t, (eeg_eo));
title('Raw EEG EO') 
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
set(gca, 'LineWidth', 1)
grid on
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
print('4_1_raw_eeg_eco','-dpng','-r300');

%% Pre-processing
%% Bandpass filtering
% highpass filter to remove low frequence drift
Wp = 8/(Fs/2);
Ws = 3/(Fs/2);
Rp = 3;
Rs = 60;
[n, Wn] = buttord(Wp,Ws,Rp, Rs);
%[b a] = ellip(n, Rp, Rs, Wn, 'high');
[b a] = butter(n, Wn, 'high');
[h,f] = freqz(b, a, N, Fs);
h = 20*log10(abs(h));
figure
%semilogx(f,h, 'b'); axis tight
subplot(2,2,1)
plot(f, h);
grid on;
xlabel('Frequency (Hz)'); ylabel('X(f)(dB)');
title('Butterworth Highpass filter');
set(gca,'xlim',[0, 10])

%apply high pass filter to original signals
eeg_ec_h = filtfilt(b, a, eeg_ec);
eeg_eo_h = filtfilt(b, a, eeg_eo);
% figure
% plot(t,eeg_ec_h)
% xlabel('Time (sec)')
% ylabel('Amplitude (\muV)')
% title('Highpass Butterworth Filtered EC')
% figure
% plot(t,eeg_eo_h)
% xlabel('Time (sec)')
% ylabel('Amplitude (\muV)')
% title('Highpass Butterworth Filtered EC')
%set(gca,'ylim',[-40, 40]) % set the limits of time in the plot

% lowpass filter to remove 50 Hz power line
Wp = 12/(Fs/2);
Ws = 50/(Fs/2);
Rp = 3;
Rs = 60;
[n, Wn] = buttord(Wp,Ws,Rp, Rs);
%[c d] = ellip(n, Rp, Rs, Wn, 'low');
[c d] = butter(n, Wn, 'low');
[h,f] = freqz(c, d, N, Fs);
h = 20*log10(abs(h));
%figure
%semilogx(f,h, 'b'); axis tight
subplot(2,2,2)
plot(f, h);
grid on;
xlabel('Frequency (Hz)'); ylabel('X(f)(dB)');
title('Butterworth Lowpass filter');
set(gca,'xlim',[0, 20]) 

%apply low pass filter to original signal
eeg_ec_l = filtfilt(c, d, eeg_ec_h);
eeg_eo_l = filtfilt(c, d, eeg_eo_h);
%figure
subplot(2,2,3)
plot(t,eeg_ec_l); axis tight
grid on;
xlabel('Time (sec)')
ylabel('Amplitude (\muV)')
title('Bandpass Filtered EC')
%figure
subplot(2,2,4)
plot(t,eeg_eo_l); axis tight
grid on;
xlabel('Time (sec)')
ylabel('Amplitude (\muV)')
title('Bandpass Filtered EO')
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
print('4_1_pre_processing_filtering','-dpng','-r300');

alpha_ec =  eeg_ec_l;
alpha_eo = eeg_eo_l;

%% Feature Selection
% Calculate PSD EC EEG filtered signal using Welch Method
% nfft = 2^nextpow2(N);
% [PS_ec,F_ec] = pwelch(detrend(eeg_ec),hamming(500),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
% [PS_eo,F_eo] = pwelch(detrend(eeg_eo),hamming(500),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000

% % alpha band power
% signal = eeg_eo;
% waveletFunction = 'sym8'%'db8' OR 'sym8' OR 'coif5' OR 'db4';
% [C,L] = wavedec(signal,7,waveletFunction);
% % Calculation the Details Vectors
% D1 = wrcoef('d',C,L,waveletFunction,1); %NOISY
% D2 = wrcoef('d',C,L,waveletFunction,2); %NOISY
% D3 = wrcoef('d',C,L,waveletFunction,3); %NOISY
% D4 = wrcoef('d',C,L,waveletFunction,4); %GAMMA
% D5 = wrcoef('d',C,L,waveletFunction,5); %BETA
% D6 = wrcoef('d',C,L,waveletFunction,6); %ALPHA
% D7 = wrcoef('d',C,L,waveletFunction,7); %THETA
% A7 = wrcoef('a',C,L,waveletFunction,7); %DELTA
% %POWER_DELTA = (sum(A8.^2))/length(A8);
% alpha_eo = D6;
% POWER_BETA_EO = (sum(D5.^2))/length(D5);
% POWER_ALPHA_EO = (sum(D6.^2))/length(D6);
% POWER_THETA_EO = (sum(D7.^2))/length(D7);
% POWER_DELTA_EO = (sum(A7.^2))/length(A7);
% 

% 
% signal = eeg_ec;
% waveletFunction = 'sym8'%'db8' OR 'sym8' OR 'coif5' OR 'db4';
% [C,L] = wavedec(signal,8,waveletFunction);
% % Calculation the Details Vectors
% D1 = wrcoef('d',C,L,waveletFunction,1); %NOISY
% D2 = wrcoef('d',C,L,waveletFunction,2); %NOISY
% D3 = wrcoef('d',C,L,waveletFunction,3); %NOISY
% D4 = wrcoef('d',C,L,waveletFunction,4); %GAMMA
% D5 = wrcoef('d',C,L,waveletFunction,5); %BETA
% D6 = wrcoef('d',C,L,waveletFunction,6); %ALPHA
% D7 = wrcoef('d',C,L,waveletFunction,7); %THETA
% A7 = wrcoef('a',C,L,waveletFunction,7); %DELTA
% %POWER_DELTA = (sum(A8.^2))/length(A8);
% alpha_ec = D6;
% POWER_BETA_EC = (sum(D5.^2))/length(D5);
% POWER_ALPHA_EC = (sum(D6.^2))/length(D6);
% POWER_THETA_EC = (sum(D7.^2))/length(D7);
% POWER_DELTA_EC = (sum(A7.^2))/length(A7);
% 

y = alpha_eo;
NFFT = 2^nextpow2(length(alpha_eo)); % Next power of 2 from length of y
    Y = fft(y,NFFT)/length(alpha_eo);
f = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure
plot(f,2*abs(Y(1:NFFT/2+1))) 
title('Single-Sided Amplitude Spectrum of Alpha Band (8-12 Hz)EO')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')

y = alpha_ec;
NFFT = 2^nextpow2(length(alpha_ec)); % Next power of 2 from length of y
Y = fft(y,NFFT)/length(alpha_ec);
f = Fs/2*linspace(0,1,NFFT/2+1);
% Plot single-sided amplitude spectrum.
figure
plot(f,2*abs(Y(1:NFFT/2+1))) 
title('Single-Sided Amplitude Spectrum of Alpha Band (8-12 Hz)EC')
xlabel('Frequency (Hz)')
ylabel('|Y(f)|')

nfft = 2^nextpow2(length(alpha_ec));
[PS_alpha_ec,F_alpha_ec] = pwelch(detrend(alpha_ec),hamming(500),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
[PS_alpha_eo,F_alpha_eo] = pwelch(detrend(alpha_eo),hamming(500),[],nfft, Fs);%default is hamming window with 50% overlap, change segment length 2000,1000, 5000
% plot(F_ec,10*log10(PS_ec),'b')
% plot(F_eo,10*log10(PS_eo),'r')
figure
box on
hold on
grid on
plot(F_alpha_ec,(PS_alpha_ec),'b')
plot(F_alpha_eo,(PS_alpha_eo),'r')
% plot(F_ec,(PS_ec),'g')
% plot(F_eo,(PS_eo),'y')
hold off
xlabel('Frequency (Hz)')
%ylabel('PSD (dB)')
ylabel('PSD (\muV^2/Hz)')
title('PSD (Welch Method)')
legend('Alpha Band PSD EC', 'Alpha Band PSD EO')%, 'PSD EC', 'PSD EO')
set(gca,'xlim',[0 20]);
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
print('4_1_alpha_band_psd','-dpng','-r300');

%% Processing to generate trials
%Sampling frequency of eeg_eo and eeg_ec is 1000 Hz
%Duration 180 seconds
%total data points = 180,000

%Framing of data with duration 1 second, break data into seconds
data_eo_sec = reshape(eeg_eo, 1000, 180);
data_ec_sec = reshape(eeg_ec, 1000, 180);
data_eo_sec_alpha = reshape(alpha_eo, 1000, 180);
data_ec_sec_alpha = reshape(alpha_ec, 1000, 180);

data_eo = zeros(200,900);
data_ec = zeros(200,900);
data_eo_alpha = zeros(200,900);
data_ec_alpha = zeros(200,900);
for i=1:180
    %subsample each second at 200 Hz
    for j=1:5
        data_eo(:,(5*(i-1))+j) = data_eo_sec(j:5:end,i);
        data_ec(:,(5*(i-1))+j) = data_ec_sec(j:5:end,i);
        data_eo_alpha(:,(5*(i-1))+j) = data_eo_sec_alpha(j:5:end,i);
        data_ec_alpha(:,(5*(i-1))+j) = data_ec_sec_alpha(j:5:end,i);
    end
end
Fs = 200;
samples = 200;
trials = 900;

%% Feature Extraction

alpha_psd_eo = zeros(900,1);
alpha_psd_ec = zeros(900,1);
alpha_power_eo = zeros(900,1);
alpha_power_ec = zeros(900,1);

nfft = 2^nextpow2(samples);
for i=1:900
    % Calculate alpha band PSD
    [PS_alpha_eo,F_alpha_eo] = pwelch(detrend(data_eo_alpha(:,i)),hamming(Fs),[],nfft, Fs);%default is hamming window with 50% overlap
    [PS_alpha_ec,F_alpha_ec] = pwelch(detrend(data_ec_alpha(:,i)),hamming(Fs),[],nfft, Fs);%default is hamming window with 50% overlap    
%     [psd_eo,F_eo] = pwelch(detrend(data_eo(:,i)),hamming(Fs),[],nfft, Fs);%default is hamming window with 50% overlap
%     [psd_ec,F_ec] = pwelch(detrend(data_ec(:,i)),hamming(Fs),[],nfft, Fs);%default is hamming window with 50% overlap
%     figure
%     box on
%     hold on
%     plot(F_ec,(psd_ec),'b')
%     plot(F_eo,(psd_eo),'r') 
%     plot(F_alpha_ec,(PS_alpha_ec),'g')
%     plot(F_alpha_eo,(PS_alpha_eo),'y')
%     hold off
%     xlabel('Frequency (Hz)')
%     %ylabel('PSD (dB)')
%     ylabel('PSD (\muV^2/Hz)')
%     title('PSD (Welch Method) 2')
%     legend('PSD EC', 'PSD EO', 'Alpha Band PSD EC', 'Alpha Band PSD EO')
    
    % Get max alpha band PSD
    alpha_psd_ec(i) = max(PS_alpha_ec);
    alpha_psd_eo(i) = max(PS_alpha_eo);
    % Calculate Alpha Power
    alpha_power_eo(i) = (sum(data_eo_alpha(:,i).^2))/length(data_eo_alpha(:,i));
    alpha_power_ec(i) = (sum(data_ec_alpha(:,i).^2))/length(data_ec_alpha(:,i));
end

%% Outlier removal use absolute distance to remove outliers greater than 3 standard deviations
Index =  abs(alpha_power_eo - median(alpha_power_eo)) > 3*std(alpha_power_eo);
count = 1;
%alpha_power_eo_temp;
for i=1:900
    if(Index(i)==0)
        alpha_power_eo_temp(count) = alpha_power_eo(i);
     count = count + 1;
    end
end
alpha_power_eo = transpose(alpha_power_eo_temp);

I =   abs(alpha_power_ec - median(alpha_power_ec)) > 3*std(alpha_power_ec);
count = 1;
%alpha_power_ec_temp;
for i=1:900
    if(Index(i)==0)
        alpha_power_ec_temp(count) = alpha_power_ec(i);
     count = count + 1;
    end
end
alpha_power_ec = transpose(alpha_power_ec_temp);

trials = min(length(alpha_power_ec), length(alpha_power_eo));
%generate training and test data
P = 1; % number of features
N_train = trials*0.80; % 80% training
N_test = trials*0.20; % 20% test

% train_data_eo = [alpha_psd_eo(1:N_train),alpha_power_eo(1:N_train)];
% test_data_eo = [alpha_psd_eo((N_train+1):trials),alpha_power_eo((N_train+1):trials)];
% 
% train_data_ec = [alpha_psd_ec(1:N_train),alpha_power_ec(1:N_train)];
% test_data_ec = [alpha_psd_ec((N_train+1):trials),alpha_power_ec((N_train+1):trials)];

train_data_eo = alpha_power_eo(1:N_train);
test_data_eo = alpha_power_eo((N_train+1):trials);

train_data_ec = alpha_power_ec(1:N_train);
test_data_ec = alpha_power_ec((N_train+1):trials);

% Plot the samples
figure
hold on; box on;
% plot(train_data_eo(:,1), train_data_eo(:,2), 'ro');
% plot(train_data_ec(:,1), train_data_ec(:,2),'bo');
plot(train_data_eo(:,1), 1:N_train, 'ro');
plot(train_data_ec(:,1), 1:N_train,'bo');
plot(test_data_eo, (N_train+1):trials,'r+');
plot(test_data_ec, (N_train+1):trials,'b+');
grid on;
title('Data Plots');
xlabel('Alpha Power (\muV^2)');
ylabel('Trial');
legend({'EO (Training)','EC (Training)', 'EO (Test)','EC (Test)'})
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
print('4_1_training_and_test_data','-dpng','-r300');
%set(gca,'xlim',[-4 4],'ylim',[-4 4])

%all data, concat training and test data/specify labels
train_samples = [train_data_eo; train_data_ec];
train_labels =[ones(N_train,1); zeros(N_train,1)];

test_samples = [test_data_eo; test_data_ec];
test_labels =[ones(N_test,1); zeros(N_test,1)];

% LDA
classout = classify(test_samples,train_samples,train_labels,'linear');
% Compute accuracy
acc = sum(classout==test_labels)/(N_test*2);
% Compute TP/TN/FP/FN and sensitivity/specificity
% EO: positive; EC: negative (case-dependent)
TP = sum((classout==test_labels)&(classout==0));
TN = sum((classout==test_labels)&(classout==1));
FP = sum((classout~=test_labels)&(classout==0));
FN = sum((classout~=test_labels)&(classout==1));
sensitivity = TP/(TP+FN);
specificity = TN/(TN+FP);
accuracy = TP+TN/(TP+TN+FP+FN);
%% Using LDA
%% Cross Validation
all_samples = [train_samples;test_samples]; % all available data
all_labels = [train_labels;test_labels]; % labels of all available data
K = 10; % K-fold CV
indices = crossvalind('Kfold',all_labels,K); % generate random
cv_sen = zeros(K:1);
cv_spec = zeros(K:1);
cv_acc = zeros(K:1);
for k = 1:K % K iterations
    cv_test_idx = find(indices == k); % indices for test samples in one trial of validation
    cv_train_idx = find(indices ~= k); % indices for training samples in one trial of validation
    cv_classout =classify(all_samples(cv_test_idx,:),all_samples(cv_train_idx,:),all_labels(cv_train_idx,:)); % classification
    % Compute TP/TN/FP/FN and sensitivity/specificity
    % EO: positive; EC: negative (case-dependent)
    TP = sum((classout==test_labels)&(classout==0));
    TN = sum((classout==test_labels)&(classout==1));
    FP = sum((classout~=test_labels)&(classout==0));
    FN = sum((classout~=test_labels)&(classout==1));
    cv_sen(k) = TP/(TP+FN);
    cv_spec(k) = TN/(TN+FP);
    cv_acc(k) = sum(cv_classout==all_labels(cv_test_idx,:))/(numel(cv_classout)); % compute classification accuracy
end
cv_acc_avg = mean(cv_acc); % averaged accuracy
cv_spec_avg = mean(cv_spec); % averaged accuracy
cv_sen_avg = mean(cv_sen); % averaged accuracy

%% using SVM
%% Cross Validation
all_samples = [train_samples;test_samples]; % all available data
all_labels = [train_labels;test_labels]; % labels of all available data
K = 10; % K-fold CV
indices = crossvalind('Kfold',all_labels,K); % generate random
cv_sen = zeros(K:1);
cv_spec = zeros(K:1);
cv_acc = zeros(K:1);
for k = 1:K % K iterations
    cv_test_idx = find(indices == k); % indices for test samples in one trial of validation
    cv_train_idx = find(indices ~= k); % indices for training samples in one trial of validation
    svmStruct = svmtrain(all_samples(cv_train_idx,:),all_labels(cv_train_idx,:));
    cv_classout = svmclassify(svmStruct,all_samples(cv_test_idx,:));
    % Compute TP/TN/FP/FN and sensitivity/specificity
    % EO: positive; EC: negative (case-dependent)
    TP = sum((classout==test_labels)&(classout==0));
    TN = sum((classout==test_labels)&(classout==1));
    FP = sum((classout~=test_labels)&(classout==0));
    FN = sum((classout~=test_labels)&(classout==1));
    cv_sen(k) = TP/(TP+FN);
    cv_spec(k) = TN/(TN+FP);
    cv_acc(k) = sum(cv_classout==all_labels(cv_test_idx,:))/(numel(cv_classout)); % compute classification accuracy
end
cv_acc_avg_svm = mean(cv_acc); % averaged accuracy
cv_spec_avg_svm = mean(cv_spec); % averaged accuracy
cv_sen_avg_svm = mean(cv_sen); % averaged accuracy

%% ============================4.2=========================== %%
%% using SVM

alpha_power_eeg = zeros(100,1);
%loop through all trials
for i=1:100
    eeg = unknown_trials(:,i);
    % Pre-process unknown trials
    %% Bandpass filtering
    % highpass filter to remove low frequence drift
    Wp = 8/(Fs/2);
    Ws = 1/(Fs/2);
    Rp = 3;
    Rs = 60;
    [n, Wn] = buttord(Wp,Ws,Rp, Rs);
    %[b a] = ellip(n, Rp, Rs, Wn, 'high');
    [b a] = butter(n, Wn, 'high');
    [h,f] = freqz(b, a, N, Fs);

    %apply high pass filter to original signals
    eeg_h = filtfilt(b, a, eeg);

    % lowpass filter to remove 50 Hz power line
    Wp = 12/(Fs/2);
    Ws = 50/(Fs/2);
    Rp = 3;
    Rs = 60;
    [n, Wn] = buttord(Wp,Ws,Rp, Rs);
    %[c d] = ellip(n, Rp, Rs, Wn, 'low');
    [c d] = butter(n, Wn, 'low');
    [h,f] = freqz(c, d, N, Fs);

    %apply low pass filter to original signal
    eeg_l = filtfilt(c, d, eeg_h);

    alpha_eeg =  eeg_l;
    
    % Calculate Alpha Power
    alpha_power_eeg(i) = (sum(alpha_eeg.^2))/length(alpha_eeg);    
end

% Plot the samples
figure
hold on; box on;
% plot(train_data_eo(:,1), train_data_eo(:,2), 'ro');
% plot(train_data_ec(:,1), train_data_ec(:,2),'bo');
plot(train_data_eo(:,1), 1:N_train, 'ro');
plot(train_data_ec(:,1), 1:N_train,'bo');
plot(test_data_eo, (N_train+1):trials,'r+');
plot(test_data_ec, (N_train+1):trials,'b+');
plot(alpha_power_eeg, 1:length(alpha_power_eeg),'go');
grid on;
title('Data Plots');
xlabel('Alpha Power (\muV^2)');
ylabel('Trial')
legend({'EO (Training)','EC (Training)', 'EO (Test)','EC (Test)', 'Unknown Trials'})
set(gcf, 'PaperPosition', [-0.5 -0.25 10 5.5]); %Position the plot further to the left and down. Extend the plot to fill entire paper.
set(gcf, 'PaperSize', [5 5]); %Keep the same paper size
print('4_1_training_test_unknown_data','-dpng','-r300');

%% SVM
all_samples = [train_samples;test_samples;]; % all available data
all_labels = [train_labels;test_labels]; % labels of all available data

svmStruct = svmtrain(all_samples,all_labels);
class_labels = svmclassify(svmStruct,alpha_power_eeg);

%save MAT file
save assg_aaron_sheshan.mat class_labels