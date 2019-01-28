function [X, y] = readmnist(imgfilename, labfilename)

% open image file
f = fopen(imgfilename, 'r', 'ieee-be');

% read header
magic = fread(f, [1, 1], '*uint32');
if ~isequal(magic, uint32(2051))
    fclose(f);
    error('readmnist:badMagic', 'Invalid Magic token in image file.');
end

% open label file
l = fopen(labfilename, 'r', 'ieee-be');
magic = fread(l, [1, 1], '*uint32');
if ~isequal(magic, uint32(2049))
    fclose(f);
    fclose(l);
    error('readmnist:badMagic', 'Invalid Magic token in label file.');
end

% number of items and size
numitems = fread(f, [1, 1], 'uint32=>double');
imgsize = fread(f, [1, 2], 'uint32=>double');
numlabs = fread(l, [1, 1], 'uint32=>double');
if numitems ~= numlabs
    fclose(f);
    fclose(l);
    error('readmnist:incosistentFiles', 'Inconsistent files.');
end

% read images
try
    X = fread(f, [prod(imgsize), numitems], '*uint8');
    X = permute(reshape(X, [imgsize, numitems]), [3, 2, 1]);
    y = fread(l, [numitems, 1], 'uint8=>double');
catch e
    fclose(f);
    fclose(l);
    rethrow(e);
end
fclose(f);
fclose(l);
