function stats = fit_raw_channel_normalization(raw, features, indices)
% Estimate per-channel normalization using source_train only.

indices = indices(:);
channelSum = zeros(4,1);
channelSumSq = zeros(4,1);
count = 0;
for i = 1:numel(indices)
    row = indices(i);
    x = double(raw{features.y_class(row), features.y_trial(row)});
    channelSum = channelSum + sum(x,2);
    channelSumSq = channelSumSq + sum(x.^2,2);
    count = count + size(x,2);
end
stats.mean = channelSum / count;
variance = max(channelSumSq / count - stats.mean.^2, eps);
stats.std = sqrt(variance);
stats.fitIndices = indices;
end

