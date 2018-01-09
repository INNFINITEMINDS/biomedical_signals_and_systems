function [y, h] = wiener_filter(x, Px0, Pn)
% Wiener Filter (FFT-based)
% // Input // %
% x:        input signal (with noise)
% Px0:      spectrum of orignal clean signal (no noise)
% Pn:       spectrum of variance of noise (i.i.d.)

% // Output // %
% y:        response/output
% h:        coefficients of the Wiener filter

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% by Zhiguo Zhang, 10/2014
% ========================================================  %

%% Operation
H = Px0./(Px0 + Pn); % Fourier transform of filter h
% compute convolution
Fx = fft(x);
y = real( ifft(Fx.*H) );
h = fftshift( ifft(H) );
