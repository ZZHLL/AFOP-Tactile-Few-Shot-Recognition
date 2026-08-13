function data = load_project_data(cfg, loadRaw)
% Load the prepared dataset, features, labels, and split.

if nargin < 2
    loadRaw = true;
end

required = {cfg.input.labels, cfg.input.features};
if loadRaw
    required{end+1} = cfg.input.dataset;
end
for i = 1:numel(required)
    assert(isfile(required{i}), 'Missing required input: %s', required{i});
end

labelsFile = load(cfg.input.labels, 'labels_table');
featureFile = load(cfg.input.features, 'features_with_labels');
data.labelsTable = labelsFile.labels_table;
data.features = featureFile.features_with_labels;
data.fold = make_closedset_splits(data.features, ...
    cfg.fold.trainRatio, cfg.fold.innerTrainRatio, cfg.seed.fold);

if loadRaw
    rawFile = load(cfg.input.dataset, 'dataset3_jittered');
    data.raw = rawFile.dataset3_jittered;
else
    data.raw = [];
end

validate_project_data(data, cfg, loadRaw);
end
