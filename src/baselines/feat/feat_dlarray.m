function value = feat_dlarray(value, device)
% Convert C x T x B raw trials to 1 x T x C x B tensors.

value = permute(single(value), [4 2 1 3]);
if device == "gpu"
    value = gpuArray(value);
end
value = dlarray(value, 'SSCB');
end
