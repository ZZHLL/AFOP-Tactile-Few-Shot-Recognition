function [params, trailingAverage, trailingAverageSq] = adamupdate_mamlpp(params, gradients, trailingAverage, trailingAverageSq, iteration, learningRate)
% Apply Adam to each explicit MAML parameter.

names = fieldnames(params);
if isempty(trailingAverage)
    trailingAverage = struct();
    trailingAverageSq = struct();
    for i = 1:numel(names)
        trailingAverage.(names{i}) = [];
        trailingAverageSq.(names{i}) = [];
    end
end
for i = 1:numel(names)
    name = names{i};
    [params.(name), trailingAverage.(name), trailingAverageSq.(name)] = ...
        adamupdate(params.(name), gradients.(name), trailingAverage.(name), ...
        trailingAverageSq.(name), iteration, learningRate);
end
end
