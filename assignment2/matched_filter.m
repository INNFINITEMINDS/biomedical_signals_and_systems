function [y, h] = matched_filter(x, template, noise_var)
% Matched Filter
% // Input // %
% x:            input signal
% template:     template
% noise_var:	variance of noise (i.i.d.)

% // Output // %
% y:            response/output
% h:            coefficients of the matched filter

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% by Zhiguo Zhang, 10/2014
% ========================================================  %

%% Operation
R = noise_var*eye(length(template)); % covariance matrix of noise
h = (inv(R)/sqrt(template'*inv(R)*template))*template; 
h = h(length(template):-1:1);
y = filtfilt(h,1,x);
