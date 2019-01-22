function [c, cg] = rcost(t, x, y)
%RCOST  Regression cost function (1/2m .* sum-of-squared-error)
%   C = RCOST(T, X, Y) computes the cost as (1 / (2*m)) * sum(E .^ 2),
%   with E being the simple error term (y

% compute error as prediction - y
e = rpred(t, x) - y(:);

% compute cost
c = (1 / (2 * numel(e))) * sum(e .* e);

% also compute gradient?
if nargout > 1
    cg = (1 / numel(y)) .* (x' * e);
end
