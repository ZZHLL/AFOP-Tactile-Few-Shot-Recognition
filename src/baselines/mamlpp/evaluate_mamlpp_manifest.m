function result = evaluate_mamlpp_manifest(params, raw, features, manifest, normalization, cfg)
% Evaluate raw-signal MAML++ on a stored test manifest.

template = struct('N_WAY',[],'K_SHOT',[],'Q_SHOT',[],'episodes',[], ...
    'acc_mean',[],'acc_std',[],'acc_ci95',[],'adapt_ms_median',[], ...
    'infer_ms_per_sample_median',[],'episode_accuracy',[]);
result = repmat(template, size(manifest.combos,1), 1);
for comboIdx = 1:size(manifest.combos,1)
    entries = manifest.entries{comboIdx};
    accuracy = zeros(numel(entries),1);
    adaptationMs = zeros(numel(entries),1);
    inferenceMs = zeros(numel(entries),1);
    queryCount = zeros(numel(entries),1);
    for ep = 1:numel(entries)
        episode = raw_episode_from_manifest(raw, features, entries(ep), normalization);
        [probability, adaptationMs(ep), inferenceMs(ep)] = ...
            mamlpp_adapt_predict(params, episode, cfg);
        [~, localPrediction] = max(gather(extractdata(probability)), [], 1);
        predictions = episode.classIds(localPrediction(:));
        accuracy(ep) = mean(predictions(:) == episode.queryGlobalLabels(:));
        queryCount(ep) = numel(predictions);
    end
    combo = manifest.combos(comboIdx,:);
    item = template;
    item.N_WAY = combo(1); item.K_SHOT = combo(2); item.Q_SHOT = combo(3);
    item.episodes = numel(entries);
    item.acc_mean = mean(accuracy); item.acc_std = std(accuracy);
    item.acc_ci95 = 1.96*item.acc_std/sqrt(numel(accuracy));
    item.adapt_ms_median = median(adaptationMs);
    item.infer_ms_per_sample_median = median(inferenceMs ./ queryCount);
    item.episode_accuracy = accuracy;
    result(comboIdx) = item;
end
end
