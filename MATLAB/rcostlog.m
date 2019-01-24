function [c, cg] = rcostlog(t, x, y, l)
%RCOSTLOG  Regression cost function (1/2m .* sum-of-squared-error)
%   C = RCOSTLOG(T, X, Y) computes the cost as (1 / (2*m)) * sum(E .^ 2),
%   with E being the simple error term (y

% compute error as prediction - y
p = rpredlog(t, x);
p(p == 0) = eps;
p(p == 1) = (1 - eps);
e = p - y(:);
ne = numel(e);

% compute log terms
log1 = log(p);
log0 = log(1 - p);

% without lambda
if nargin < 4 || ~isa(l, 'double') || numel(l) > 1 || l <= 0
    
    % compute cost
    c = (-1 / ne) * (log1(:)' * y(:) + log0(:)' * (1 - y(:)));

    % also compute gradient?
    if nargout > 1
        cg = (1 / ne) .* (x' * e);
    end
    
% with lambda
else
    
    % compute cost
    c = (-1 / ne) * (log1(:)' * y(:) + log0(:)' * (1 - y(:))) + ...
        (l / (2 * ne)) * sum(t(2:end) .* t(2:end));

    % also compute gradient?
    if nargout > 1
        cg = (1 / ne) .* ((x' * e) + l .* ([0; ones(numel(t)-1, 1)] .* t(:)));
    end
end
