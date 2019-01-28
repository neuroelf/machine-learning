function [J, grad, p] = allcost(theta, X, Y, lambda, tshapes)

% convert shapes?
if nargin > 4 && (isa(theta, 'double') || isa(theta, 'single') || isa(theta, 'gpuArray')) && iscell(tshapes)
    ot = theta;
    theta = cell(numel(tshapes), 1);
    ti = 1;
    for tc = 1:numel(theta)
        theta{tc} = reshape(ot(ti:ti+prod(tshapes{tc})-1), tshapes{tc});
        ti = ti + prod(tshapes{tc});
    end
elseif isa(theta, 'double')
    theta = {theta};
end

% easier with Y transposed
if size(theta{end}, 1) == size(Y, 2)
    Y = Y';
end

% number of samples (scaling)
m = size(Y, 2);

% activation values and initialize deltas
Xt = cell(numel(theta)+1, 1);
d = cell(size(Xt));
grad = cell(size(theta));

% transpose X if necessary
if ~any((size(X, 1) + [0, 1]) == size(theta{1}, 2)) && ...
    any((size(X, 2) + [0, 1]) == size(theta{1}, 2))
    Xt{1} = X';
else
    Xt{1} = X;
end

% theta squared sum
tss = 0;

% forward prop
for tc = 1:numel(theta)
    if size(Xt{tc}, 1) < size(theta{tc}, 2)
        Xt{tc} = [ones(1, size(Xt{tc}, 2)); Xt{tc}];
    end
    Xt{tc+1} = sigmoid(theta{tc} * Xt{tc});
    tsq = theta{tc} .* theta{tc};
    if ~iscell(lambda)
        tss = tss + lambda .* sum(sum(tsq(:, 2:end)));
    else
        if isequal(size(lambda{tc}), size(tsq))
            tss = tss + sum(sum(lambda{tc}(:, 2:end) .* tsq(:, 2:end)));
        else
            tss = tss + sum(sum(lambda{tc} .* tsq(:, 2:end)));
        end
    end
end

% output layer (predictions)
p = Xt{end};

% sanity checks
p(p == 0) = eps;
p(p == 1) = (1 - eps);

% compute output layer delta (error)
d{end} = p - Y;

% compute cost function
J = (-1 / m) * sum(sum(Y .* log(p) + (1 - Y) .* log(1 - p))) + ...
    (0.5 / m) * tss;

% done?
if nargout < 2
    return;
end

% back-prop
for tc = numel(theta):-1:1
    if tc > 1
        d{tc} = theta{tc}' * d{tc+1} .* (Xt{tc} .* (1 - Xt{tc}));
        d{tc}(1, :) = [];
    end
    g = (1 / m) .* (d{tc + 1} * Xt{tc}');
    if ~iscell(lambda)
        l = ((1 / m) .* lambda) .* ones(size(theta{tc}));
    else
        l = (1 / m) .* lambda{tc};
    end
    l(:, 1) = 0;
    if isequal(size(g), size(theta{tc}))
        grad{tc} = g + l .* theta{tc};
    else
        grad{tc} = g(2:end, :) + l .* theta{tc};
    end
end

% unroll
if nargin > 4 && iscell(tshapes)
    for tc = 1:numel(theta)
        grad{tc} = grad{tc}(:);
    end
    grad = cat(1, grad{:});
end

% predictions?
if nargout > 2
    [~, p] = max(p, [], 1);
    p = p(:);
end
