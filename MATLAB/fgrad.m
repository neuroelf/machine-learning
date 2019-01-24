function tg = fgrad(f, t, varargin)
%FGRAD  compute functional gradient for f with parameters t, for x->y

% four inputs
if nargin < 2
    error('fgrad:invalidNumberOfInputs', 'FGRAD requires at least 2 inputs.');
end

% f must be a function handle
if ~isa(f, 'function_handle')
    error('fgrad:invalidFunctionHandle', 'F must be a function handle.');
end

% t must be numeric
if ~isa(t, 'double') || isempty(t)
    error('fgrad:invalidTheta', 'T must be a double vector.');
end

% for each t compute gradient
tt = t;
g = 2 .^ max(-20, ceil(log2(abs(t) + eps)) - 10);
tg = t - g;
tu = t + g;
for tc = 1:numel(t)
    
    % step in both directions
    tt(tc) = tg(tc);
    rcd = f(tt, varargin{:});
    tt(tc) = tu(tc);
    rcu = f(tt, varargin{:});
    tt(tc) = t(tc);
    
    % compute gradient
    tg(tc) = (rcu - rcd);
end

% scale gradient
tg = tg ./ (2 .* g);
