function nodes = raw_channel_node_cells_from_indices(raw, features, indices, stats, temporalBins, modalityIds)
% Build four graph nodes per trial from normalized temporal summaries.
% Each node contains binned mean, binned RMS, channel id, and modality id.

indices = indices(:);
nodes = cell(numel(indices),1);
channelIdentity = eye(4, 'single');
modalityIdentity = eye(2, 'single');
for i = 1:numel(indices)
    row = indices(i);
    x = single(raw{features.y_class(row), features.y_trial(row)});
    x = normalize_raw_tensor(x, stats);
    edges = round(linspace(1, size(x,2)+1, temporalBins+1));
    means = zeros(4, temporalBins, 'single');
    rmsValues = zeros(4, temporalBins, 'single');
    for b = 1:temporalBins
        segment = x(:, edges(b):edges(b+1)-1);
        means(:,b) = mean(segment,2);
        rmsValues(:,b) = sqrt(mean(segment.^2,2));
    end
    nodeMatrix = zeros(2*temporalBins + 6, 4, 'single');
    for channel = 1:4
        nodeMatrix(:,channel) = [means(channel,:)'; rmsValues(channel,:)'; ...
            channelIdentity(:,channel); modalityIdentity(:,modalityIds(channel))];
    end
    nodes{i} = nodeMatrix';
end
end
