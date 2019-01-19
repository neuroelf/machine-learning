function [t, ts] = gdesc(f, t, x, y, l)

% if learning rate not specified
if nargin < 5 || ~isa(l, 'double') || numel(l) ~= 1 || isinf(l) || isnan(l) || l <= 0 || l > 1
    l = 0.1;
end

% maximum number of iterations
maxiter = 5000;

% compute initial cost
c0 = f(t, x, y);
seps = 4 * eps;
ts = NaN .* zeros(maxiter, numel(t));

% keep iterating
it = 1;
while it <= maxiter
    
    % compute gradient
    tg = fgrad(f, t, x, y);
    
    % update t
    for tc = 1:numel(t)
        t{tc} = t{tc} - l * tg{tc};
    end
    ts(it, :) = [t{:}];
    
    % compute new cost
    c = f(t, x, y);
    
    % if difference < 4 * eps, break
    if abs(c0 - c) < seps
        break;
    end
    c0 = c;
    it = it + 1;
end
