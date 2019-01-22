function p = rpredlog(t, x)
%RPREDLOG  Compute logistic regression prediction values.
%   P = RPREDLOG(T, X, Y) computes (1 / (1 + exp(-(X * T)))) as a vector.
p = 1 ./ (1 + exp(-(x * t(:))));
