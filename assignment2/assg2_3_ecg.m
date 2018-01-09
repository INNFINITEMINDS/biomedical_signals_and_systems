clear all; 
close all;

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% Assignment 2_3
% by Zhiguo Zhang, 10/2013

%% ============================2.1=========================== %%
% Matched Filter
% load signal
load assg2_ecg
N = length(signal);
t = [1:N]/Fs;

%Perform Matched Filter
%x(300:1) = -1773.1505;
[y, h] = matched_filter(signal, template, noise_var);
%Display Results
figure
subplot(3,1,1)
plot(t,signal,'b');
%set(gca,'xlim',[1 N],'ylim',[-100,200])
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
legend('Noisy Signal')
subplot(3,1,2)
plot([1:length(template)]/Fs,template,'k');
set(gca,'xlim',[0 length(template)/Fs])%,'ylim',[-100,200])
xlabel('Time (s)');
ylabel('Magnitude')
legend('Template')
subplot(3,1,3)
plot(t,y,'r');
%(gca,'xlim',[1 N],'ylim',[0,500])
xlabel('Time (s)')
ylabel('Magnitude')
legend('Output')

y = y - 1581.753;
figure
grid on
hold on; box on;
plot(t,signal,'b');
plot(t,y,'r');
legend('Noisy','Filtered');
ylabel('Amplitude (\muV)');
xlabel('Time (s)');
title('Comparison Noisy vs Filtered');
hold off; box off; grid off;
%print('matched_filter_3_1','-dpng','-r300');

%% ============================2.2=========================== %%
% Weiner Filter
P_clean_signal = abs(fft(clean_signal)).^2; %spectrum of clean signal
P_noise = N*noise_var; %spctrum of noise
[y, h] = wiener_filter(signal, P_clean_signal, P_noise);

%Display Results
figure
subplot(211)
plot(t,signal,'b');
xlabel('Time (s)')
ylabel('Amplitude (\muV)')
%set(gca,'xlim',[1 N],'ylim',[-10,110])
legend('Noisy Signal')
subplot(212)
plot(t,y,'r');
xlabel('Time (s)')
ylabel('Magnitude')
%set(gca,'xlim',[1 N],'ylim',[-10,110])
legend('Output')
%print('weiner_filter_3_2','-dpng','-r300');

y = y - 735.4;
figure
grid on
hold on; box on;
plot(t,signal,'b');
plot(t,y,'r');
legend('Noisy','Filtered');
ylabel('Amplitude (\muV)');
xlabel('Time (s)');
title('Comparison Noisy vs Filtered');