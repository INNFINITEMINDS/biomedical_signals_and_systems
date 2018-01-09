function [c, y, e] = lms_filter(x, d, L, mu)
% Least Mean Squares
% // Input // %
% x:        input to the filter
% d:        desired response
% L:        length of the filter
% mu:       step-size

% // Output // %
% c:        coefficients of the filter
% y:        response
% e:        error between d and y

%% ======================================================= %%
% ELEC 6081 Biomedical Signals and Systems
% by Zhiguo Zhang, 09/2013
% ========================================================  %

%% Initialization
N = length(x); % data length
c0 = zeros(L,1); % initial value of the coefficient
x_pad = padarray(x,L-1,0,'pre'); % pad x with zeros so that the fiter can work at the very begining

%% Iteration
for n=1:N
    if n==1
        cn = c0;
    else
        cn = c(:,n-1);
    end
    X = x_pad(n+L-1:-1:n);
    y(n) = cn.'*X;
    e(n) = d(n) - y(n);
    c(:,n) = cn + mu*conj(e(n))*X;
end