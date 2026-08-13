function validate_project_data(data, cfg, checkRaw)
% Validate the prepared data and split structure.

if nargin < 3
    checkRaw = true;
end

F = data.features;
assert(isequal(size(F.X_all), [2160 386]), 'Expected X_all to be 2160 x 386.');
assert(numel(F.y_class) == 2160 && numel(F.y_trial) == 2160, ...
    'Feature labels must contain 2160 rows.');
assert(~any(F.bad_mask), 'features_with_labels contains invalid rows.');
assert(all(ismember(unique(F.y_class), 1:cfg.data.numClasses)), 'Unexpected class ids.');
assert(all(ismember(unique(F.y_trial), 1:cfg.data.trialsPerClass)), 'Unexpected trial ids.');

fold = data.fold;
parts = [fold.train_all(:); fold.test_all(:)];
assert(numel(unique(parts)) == 2160, 'Closed-set train/test split is not a partition.');
assert(isempty(intersect(fold.train_all, fold.test_all)), 'Train/test leakage detected.');
assert(isempty(intersect(fold.source_train, fold.source_val)), 'Source train/val leakage detected.');
assert(all(ismember(fold.source_train, fold.train_all)), 'source_train is outside train_all.');
assert(all(ismember(fold.source_val, fold.train_all)), 'source_val is outside train_all.');
assert(numel(fold.train_all) == 1080 && numel(fold.test_all) == 1080, ...
    'Expected 1080 train and 1080 test rows.');
assert(numel(fold.source_train) == 864 && numel(fold.source_val) == 216, ...
    'Expected 864 source_train and 216 source_val rows.');

for c = 1:cfg.data.numClasses
    assert(nnz(F.y_class(fold.train_all) == c) == 30, 'Class %d train count changed.', c);
    assert(nnz(F.y_class(fold.test_all) == c) == 30, 'Class %d test count changed.', c);
    assert(nnz(F.y_class(fold.source_train) == c) == 24, 'Class %d source_train count changed.', c);
    assert(nnz(F.y_class(fold.source_val) == c) == 6, 'Class %d source_val count changed.', c);
end

if checkRaw
    assert(isequal(size(data.raw), [cfg.data.numClasses cfg.data.trialsPerClass]), ...
        'Raw dataset must be 36 x 60 cells.');
    probeRows = [1, 360, 1080, 2160];
    for row = probeRows
        c = F.y_class(row);
        t = F.y_trial(row);
        x = data.raw{c,t};
        assert(isequal(size(x), [cfg.data.numChannels cfg.data.signalLength]), ...
            'Raw mapping failed at feature row %d -> {%d,%d}.', row, c, t);
        assert(all(isfinite(x(:))), 'Non-finite raw sample at {%d,%d}.', c, t);
    end
end
end
