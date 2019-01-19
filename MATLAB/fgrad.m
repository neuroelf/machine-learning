function tg = fgrad(f, t, x, y)
%FGRAD  compute functional gradient for f with parameters t, for x->y

% four inputs
if nargin ~= 4
    error('fgrad:invalidNumberOfInputs', 'FGRAD requires 4 inputs.');
end

% f must be a function handle
if ~isa(f, 'function_handle')
    error('fgrad:invalidFunctionHandle', 'F must be a function handle.');
end

% t, x must be cell, y must be numeric
if ~iscell(t)
    error('fgrad:invalidCell', 'T must be of type cell.');
end
if ~isa(y, 'double') || isempty(y)
    error('fgrad:invalidY', 'Y must be double and non-empty.');
end

% for each t compute gradient
tt = t;
tg = cell(size(t));
for tc = 1:numel(t)
    
    % depending on magnitude
    m = max(-20, ceil(log2(abs(t{tc}) + eps)) - 10);
    g = 2 ^ m;
    
    % step in both directions
    tt{tc} = t{tc} - g;
    rcd = f(tt, x, y);
    tt{tc} = t{tc} + g;
    rcu = f(tt, x, y);
    tt{tc} = t{tc};
    
    % compute gradient
    tg{tc} = (rcu - rcd) / (2 * g);
end
