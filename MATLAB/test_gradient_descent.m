function th = test_gradient_descent

% assign test values
t = {0, 0};
x = {ones(20, 1), randn(20, 1)};
y = randn(20, 1);

% run function
th = gdesc(@rcost, t, x, y, 0.1);
