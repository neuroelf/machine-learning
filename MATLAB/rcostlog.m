function [c, cg] = rcostlog(t, x, y)
%RCOSTLOG  Regression cost function (1/2m .* sum-of-squared-error)
%   C = RCOSTLOG(T, X, Y) computes the cost as (1 / (2*m)) * sum(E .^ 2),
%   with E being the simple error term (y

% compute error as prediction - y
p = rpredlog(t, x);
p(p == 0) = eps;
p(p == 1) = (1 - eps);
e = p - y(:);

% compute log terms
log1 = -log(p);
log0 = -log(1 - p);

% compute cost
c = (1 / (numel(e))) * (log1(:)' * y(:) + log0(:)' * (1 - y(:)));

% also compute gradient?
if nargout > 1
    cg = (1 / numel(y)) .* (x' * e);
end
