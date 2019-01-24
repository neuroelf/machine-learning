function [c, cg] = rcost(t, x, y, l)
%RCOST  Regression cost function (1/2m .* sum-of-squared-error)
%   C = RCOST(T, X, Y) computes the cost as (1 / (2*m)) * sum(E .^ 2),
%   with E being the simple error term (y

% compute error as prediction - y
e = rpred(t, x) - y(:);
ne = numel(e);

% without lambda
if nargin < 4 || ~isa(l, 'double') || numel(l) > 1 || l <= 0

    % compute cost
    c = (1 / (2 * ne)) * (e' * e);

    % also compute gradient?
    if nargout > 1
        cg = (1 / ne) .* (x' * e);
    end
    
% with lambda
else
    
    % compute cost
    c = (1 / (2 * ne)) * (e' * e + l * sum(t(2:end) .* t(2:end)));
    
    % also compute gradient
    if nargout > 1
        cg = (1 / ne) .* ((x' * e) + l .* ([0; ones(numel(t)-1, 1)] .* t(:)));
    end
end
