function [c, y, e] = rls_filter(x, d, L, lambda, delta)
% Recursive Least Squares Filter
% // Input // %
% x:        input to the filter
% d:        desired response
% L:        length of the filter
% lambda:   forgetting factor
% delta:    parameter to initialize P (inverse of covariance)

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
P0 = 1/delta*eye(L); % initial value of the inverse covariance
x_pad = padarray(x,L-1,0,'pre'); % pad x with zeros so that the fiter can work at the very begining

%% Iteration
for n=1:N
    if n==1
        cn = c0;
        Pn = P0;
    else
        cn = c(:,n-1);
        Pn = P(:,:,n-1);
    end
    X = x_pad(n+L-1:-1:n);
    y(n) = cn.'*X;
    e(n) = d(n) - y(n);
    K(:,n) = Pn*conj(X)/(lambda+X.'*Pn*conj(X));
    P(:,:,n) = inv(lambda)*(eye(L)-K(:,n)*X.')*Pn;
    c(:,n) = cn + K(:,n)*e(n);
end
