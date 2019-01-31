% settings
l1 = [25, 401];
l2 = [10, l1(1)+1];
shapes = {l1, l2};

% options
options = optimset('GradObj', 'on', 'MaxIter', 100);
lambda = 0.1;

% load training and testing datasets
fid = fopen('../python/mnist_5000_20_20.bin', 'r');
X = (1 / 255) .* fread(fid, [5000, 400], 'uint8=>double')';
fclose(fid);
fid = fopen('../python/mnist_5000_20_20_lab.bin', 'r');
y = fread(fid, [5000, 1], 'uint8=>double');
fclose(fid);
Y = zeros(10, numel(y));
Y(y + (0:10:49990)') = 1;

% generate random t (with fixed values), [25, 401]; [10, 26]
rand('twister', 0);
t = 0.5 - rand(prod(l1) + prod(l2), 1);

% compute initial cost
[J, g, p] = allcost(t, X, Y, lambda, shapes);
disp(J)

[J, Jg, p] = allcost(final_t, Xtst, Ytst, lambda, {l1,l2});
disp(1 - sum(p == ytst) / numel(ytst));
n = histcounts2(ytst, p);
disp(n);
