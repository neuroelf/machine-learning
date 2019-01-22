function p = rpred(t, x)
%RPRED  Compute linear regression prediction values.
%   P = RPRED(T, X, Y) computes X * T as a vector.
p = x * t(:);
