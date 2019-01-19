function c = rcost(t, x, y)
%RCOST  regression cost function (1/2m .* sum-of-squared-error)

% compute prediction
p = cat(2, x{:}) * cat(1, t{:});

% compute error
e = y - p;

% compute cost
c = (1 / (2 * numel(e))) * sum(e .* e);
