function result = evaluate_support_only_raw_model_checkpointed(family, raw, ...
    features, fold, manifest, cfg, checkpointPath, progressPath, fullComboIndex)
% Evaluate one support-only combination with episode-level checkpointing.

family = string(family);
assert(any(family == ["tactile_transformer", "channel_gat"]), ...
    'Unsupported support-only family: %s', family);
assert(size(manifest.combos, 1) == 1, 'Evaluate one combination per run.');
assert(isempty(intersect(fold.train_all(:), manifest.allowedIdx(:))), ...
    'Support-only evaluation must not use source rows.');
assert(all(ismember(manifest.allowedIdx(:), fold.test_all(:))), ...
    'Support-only manifest contains rows outside test_all.');

combo = manifest.combos(1, :);
entries = manifest.entries{1};
state = struct('family', char(family), 'combo', combo, ...
    'fullComboIndex', fullComboIndex, 'completed', 0, ...
    'accuracy', nan(numel(entries), 1), ...
    'adaptationMs', nan(numel(entries), 1), ...
    'inferenceMs', nan(numel(entries), 1), ...
    'queryCount', nan(numel(entries), 1));
if isfile(checkpointPath)
    loaded = load(checkpointPath, 'state');
    assert(string(loaded.state.family) == family, 'Checkpoint family mismatch.');
    assert(isequal(loaded.state.combo, combo), 'Checkpoint combo mismatch.');
    assert(loaded.state.fullComboIndex == fullComboIndex, ...
        'Checkpoint seed index mismatch.');
    state = loaded.state;
    append_progress(progressPath, sprintf('[resume] %s %dw%ds at %d/%d', ...
        family, combo(1), combo(2), state.completed, numel(entries)));
end

for ep = state.completed + 1:numel(entries)
    entry = entries(ep);
    supportRows = reshape(entry.supportIdx', [], 1);
    queryRows = reshape(entry.queryIdx', [], 1);
    assert(isempty(intersect(supportRows, queryRows)), 'Support/query overlap.');
    normalization = fit_raw_channel_normalization(raw, features, supportRows);
    if family == "tactile_transformer"
        supportInput = raw_sequence_cells_from_indices(raw, features, ...
            supportRows, normalization);
        queryInput = raw_sequence_cells_from_indices(raw, features, ...
            queryRows, normalization);
        epochs = cfg.transformer.supportOnlyEpochs;
        learningRate = cfg.transformer.supportOnlyLR;
        netBuilder = @() build_tactile_transformer(cfg, combo(1));
    else
        supportInput = raw_channel_node_cells_from_indices(raw, features, ...
            supportRows, normalization, cfg.gat.temporalBins, ...
            cfg.preprocess.modalityIds);
        queryInput = raw_channel_node_cells_from_indices(raw, features, ...
            queryRows, normalization, cfg.gat.temporalBins, ...
            cfg.preprocess.modalityIds);
        epochs = cfg.gat.supportOnlyEpochs;
        learningRate = cfg.gat.supportOnlyLR;
        netBuilder = @() build_channel_gat(cfg, combo(1));
    end

    localSupport = repelem((1:combo(1))', combo(2));
    localQuery = repelem((1:combo(1))', combo(3));
    supportLabels = categorical(localSupport, 1:combo(1));
    rng(cfg.seed.eval + 100000*double(family == "channel_gat") + ...
        1000*fullComboIndex + ep, 'twister');
    start = tic;
    net = netBuilder();
    options = trainingOptions('adam', 'InitialLearnRate', learningRate, ...
        'MaxEpochs', epochs, 'MiniBatchSize', numel(supportRows), ...
        'Shuffle', 'every-epoch', 'ExecutionEnvironment', char(cfg.device), ...
        'Verbose', false, 'Plots', 'none');
    net = trainnet(supportInput, supportLabels, net, 'crossentropy', options);
    state.adaptationMs(ep) = toc(start)*1000;
    start = tic;
    metrics = evaluate_source_classifier(net, queryInput, localQuery, ...
        cfg.device, min(numel(queryRows), 64));
    state.inferenceMs(ep) = toc(start)*1000;
    state.accuracy(ep) = metrics.accuracy;
    state.queryCount(ep) = numel(queryRows);
    state.completed = ep;

    message = sprintf('[%s] %dw%ds episode %d/%d | running %.2f%% | train %.2f s', ...
        family, combo(1), combo(2), ep, numel(entries), ...
        100*mean(state.accuracy(1:ep)), state.adaptationMs(ep)/1000);
    fprintf('%s\n', message);
    append_progress(progressPath, message);
    if mod(ep, 5) == 0 || ep == numel(entries)
        save_checkpoint(checkpointPath, state);
    end
end

accuracy = state.accuracy;
result = struct('N_WAY', combo(1), 'K_SHOT', combo(2), ...
    'Q_SHOT', combo(3), 'episodes', numel(entries), ...
    'acc_mean', mean(accuracy), 'acc_std', std(accuracy), ...
    'acc_ci95', 1.96*std(accuracy)/sqrt(numel(accuracy)), ...
    'adapt_ms_median', median(state.adaptationMs), ...
    'infer_ms_per_sample_median', ...
        median(state.inferenceMs ./ state.queryCount), ...
    'episode_accuracy', accuracy, 'fullComboIndex', fullComboIndex);
end

function save_checkpoint(path, state)
temporaryPath = [path '.tmp.mat'];
save(temporaryPath, 'state', '-v7.3');
movefile(temporaryPath, path, 'f');
end

function append_progress(path, message)
fid = fopen(path, 'a');
assert(fid >= 0, 'Unable to open progress log: %s', path);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s | %s\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), message);
end
