function result = evaluate_feat_manifest(encoderNet, adapterNet, raw, features, manifest, normalization, hp, cfg)
% Evaluate a shot-specific FEAT model on stored episodes.

template = struct('N_WAY',[],'K_SHOT',[],'Q_SHOT',[],'episodes',[], ...
    'acc_mean',[],'acc_std',[],'acc_ci95',[],'adapt_ms_median',[], ...
    'infer_ms_per_sample_median',[],'episode_accuracy',[]);
result = repmat(template,size(manifest.combos,1),1);
for comboIdx = 1:size(manifest.combos,1)
    assert(manifest.combos(comboIdx,2) == hp.shot, ...
        'Shot-specific FEAT model received a mismatched combination.');
    entries = manifest.entries{comboIdx};
    accuracy = zeros(numel(entries),1);
    adaptationMs = zeros(numel(entries),1);
    inferenceMs = zeros(numel(entries),1);
    queryCount = zeros(numel(entries),1);
    for ep = 1:numel(entries)
        episode = raw_episode_from_manifest(raw,features,entries(ep),normalization);
        [probability,adaptationMs(ep),inferenceMs(ep)] = ...
            feat_predict(encoderNet,adapterNet,episode,hp,cfg);
        [~,localPrediction] = max(gather(extractdata(probability)),[],1);
        prediction = episode.classIds(localPrediction(:));
        accuracy(ep) = mean(prediction(:) == episode.queryGlobalLabels(:));
        queryCount(ep) = numel(prediction);
    end
    combo = manifest.combos(comboIdx,:);
    item = template;
    item.N_WAY = combo(1); item.K_SHOT = combo(2); item.Q_SHOT = combo(3);
    item.episodes = numel(entries);
    item.acc_mean = mean(accuracy); item.acc_std = std(accuracy);
    item.acc_ci95 = 1.96 * item.acc_std / sqrt(numel(accuracy));
    item.adapt_ms_median = median(adaptationMs);
    item.infer_ms_per_sample_median = median(inferenceMs ./ queryCount);
    item.episode_accuracy = accuracy;
    result(comboIdx) = item;
end
end
