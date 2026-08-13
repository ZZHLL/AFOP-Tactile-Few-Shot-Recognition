function params = move_mamlpp_parameters(params, device)
% Move explicit MAML parameters to CPU or GPU while preserving dlarray type.

names = fieldnames(params);
for i = 1:numel(names)
    value = extractdata(params.(names{i}));
    if device == "gpu"
        value = gpuArray(value);
    else
        value = gather(value);
    end
    params.(names{i}) = dlarray(value);
end
end
