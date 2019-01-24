function [t, ts, cs] = gdesc(f, opts, t, varargin)

% options
if nargin < 3 || ~isa(f, 'function_handle') || ~isa(t, 'double') || isempty(t)
    error('gdesc:badCall', 'Call requires at least three arguments.');
end
if ~isstruct(opts) || isempty(opts)
    opts = struct('learning_rate', 0.1, 'maxiter', 500);
end
if ~isfield(opts, 'learning_rate') || ~isa(opts(1).learning_rate, 'double') || ...
    isempty(opts(1).learning_rate)
    l = 0.1;
else
    l = max(0.0001, min(1, opts(1).learning_rate(1)));
end
if ~isfield(opts, 'maxiter') || ~isa(opts(1).maxiter, 'double') || ...
    isempty(opts(1).maxiter)
    maxiter = 500;
else
    maxiter = max(5, min(100000, ceil(opts(1).maxiter(1))));
end

% compute initial cost
funchasgrad = false;
try
    [c0, cg] = f(t, varargin{:});
    tg = fgrad(f, t, varargin{:});
    if isequal(size(cg), size(t)) && sum(abs(tg(:) - cg(:))) < (sqrt(eps) * numel(t))
        funchasgrad = true;
    end
catch
end
seps = 4 * eps;
ts = NaN .* zeros(maxiter, numel(t));
cs = NaN .* zeros(maxiter, 1);

% keep iterating
it = 1;
while it <= maxiter
    
    % update t
    t = t - l * tg;
    ts(it, :) = t;
    
    % compute new cost
    if funchasgrad
        [c, tg] = f(t, varargin{:});
    else
        c = f(t, varargin{:});
        tg = fgrad(f, t, varargin{:});
    end
    cs(it) = c;
    
    % if difference < 4 * eps, break
    if abs(c0 - c) < seps
        break;
    end
    c0 = c;
    it = it + 1;
end
