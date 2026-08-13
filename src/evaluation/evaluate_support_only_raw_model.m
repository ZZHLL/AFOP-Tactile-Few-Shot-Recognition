function result = evaluate_support_only_raw_model(family, raw, features, fold, manifest, cfg)
% Train a fresh non-meta model on each support set and evaluate its query set.

family = string(family);
assert(any(family == ["tactile_transformer","channel_gat"]), ...
    'Unsupported support-only family: %s', family);
assert(isempty(intersect(fold.train_all(:), manifest.allowedIdx(:))), ...
    'Support-only evaluation must not use source-train rows.');
assert(all(ismember(manifest.allowedIdx(:), fold.test_all(:))), ...
    'Support-only manifest contains rows outside test_all.');

template = struct('N_WAY',[],'K_SHOT',[],'Q_SHOT',[],'episodes',[], ...
    'acc_mean',[],'acc_std',[],'acc_ci95',[],'adapt_ms_median',[], ...
    'infer_ms_per_sample_median',[],'episode_accuracy',[]);
result = repmat(template, size(manifest.combos,1), 1);

for comboIdx = 1:size(manifest.combos,1)
    combo = manifest.combos(comboIdx,:);
    entries = manifest.entries{comboIdx};
    accuracy = zeros(numel(entries),1);
    adaptationMs = zeros(numel(entries),1);
    inferenceMs = zeros(numel(entries),1);
    queryCount = zeros(numel(entries),1);

    for ep = 1:numel(entries)
        entry = entries(ep);
        supportRows = reshape(entry.supportIdx', [], 1);
        queryRows = reshape(entry.queryIdx', [], 1);
        assert(isempty(intersect(supportRows, queryRows)), 'Support/query overlap.');

        normalization = fit_raw_channel_normalization(raw, features, supportRows);
        if family == "tactile_transformer"
            supportInput = raw_sequence_cells_from_indices(raw, features, supportRows, normalization);
            queryInput = raw_sequence_cells_from_indices(raw, features, queryRows, normalization);
            epochs = cfg.transformer.supportOnlyEpochs;
            learningRate = cfg.transformer.supportOnlyLR;
            netBuilder = @() build_tactile_transformer(cfg, combo(1));
        else
            supportInput = raw_channel_node_cells_from_indices(raw, features, supportRows, ...
                normalization, cfg.gat.temporalBins, cfg.preprocess.modalityIds);
            queryInput = raw_channel_node_cells_from_indices(raw, features, queryRows, ...
                normalization, cfg.gat.temporalBins, cfg.preprocess.modalityIds);
            epochs = cfg.gat.supportOnlyEpochs;
            learningRate = cfg.gat.supportOnlyLR;
            netBuilder = @() build_channel_gat(cfg, combo(1));
        end

        localSupport = repelem((1:combo(1))', combo(2));
        localQuery = repelem((1:combo(1))', combo(3));
        supportLabels = categorical(localSupport, 1:combo(1));

        rng(cfg.seed.eval + 100000*double(family == "channel_gat") + ...
            1000*comboIdx + ep, 'twister');
        start = tic;
        net = netBuilder();
        options = trainingOptions('adam', ...
            'InitialLearnRate',learningRate, ...
            'MaxEpochs',epochs, ...
            'MiniBatchSize',numel(supportRows), ...
            'Shuffle','every-epoch', ...
            'ExecutionEnvironment',char(cfg.device), ...
            'Verbose',false, ...
            'Plots','none');
        net = trainnet(supportInput, supportLabels, net, 'crossentropy', options);
        adaptationMs(ep) = toc(start)*1000;

        start = tic;
        metrics = evaluate_source_classifier(net, queryInput, localQuery, ...
            cfg.device, min(numel(queryRows), 64));
        inferenceMs(ep) = toc(start)*1000;
        accuracy(ep) = metrics.accuracy;
        queryCount(ep) = numel(queryRows);
    end

    item = template;
    item.N_WAY = combo(1);
    item.K_SHOT = combo(2);
    item.Q_SHOT = combo(3);
    item.episodes = numel(entries);
    item.acc_mean = mean(accuracy);
    item.acc_std = std(accuracy);
    item.acc_ci95 = 1.96*item.acc_std/sqrt(numel(accuracy));
    item.adapt_ms_median = median(adaptationMs);
    item.infer_ms_per_sample_median = median(inferenceMs ./ queryCount);
    item.episode_accuracy = accuracy;
    result(comboIdx) = item;
end
end
