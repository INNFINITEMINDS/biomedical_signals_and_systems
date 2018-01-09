function [P, S] = mwt(x, f, Fs, omega, sigma)
% Morlet wavelet transform (continuous)

% // Input // %
% x:            the original data samples (Time Points x Channels)
% xtimes:       the time axis of the original data
% Fs:           sampling rate
% omega:        a parameter to define the central frequency of Morlet wavelet
% sigma:        a parameter to define the spread of Morlet wavelet in time domain

% // Output // %
% P:            squared magnitude of MWT (scaleogram)
% S:            complex values of Morlet wavelet transform

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% by Zhiguo Zhang, 09/2013
% ========================================================  %

fprintf('Calculating Morlet Wavelet Transform ... ')

%% Pre-processing and Parameters
if size(x,2)==1; x = x.'; end
x = detrend(x); % Remove linear trends

N_F = length(f); % number of time samples
N_T = length(x); % number of frequency bins
f = f/Fs;     % normalized frequency

S = single(zeros(N_F,N_T));

%% Morlet wavelet transform
L_hw = N_T; % filter length
for fi=1:N_F
    scaling_factor = omega./f(fi);
    u = (-[-L_hw:L_hw])./scaling_factor;
    hw = sqrt(1/scaling_factor)*exp(-(u.^2)/(2*sigma.^2)).* exp(1i*2*pi*omega*u);
    S_full = conv(x,conj(hw));
    S(fi,:) = S_full(L_hw+1:L_hw+N_T);
end
P = abs(S).^2;

fprintf('Done!\n')