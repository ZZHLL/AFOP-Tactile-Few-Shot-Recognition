function result = evaluate_feat_plain_manifest(encoderNet, raw, features, manifest, normalization, temperature, cfg)
% Evaluate an encoder with plain prototypes on one or more combinations.

template = struct('N_WAY',[],'K_SHOT',[],'Q_SHOT',[],'episodes',[], ...
    'acc_mean',[],'acc_std',[],'acc_ci95',[],'episode_accuracy',[]);
result = repmat(template,size(manifest.combos,1),1);
for comboIndex = 1:size(manifest.combos,1)
    entries = manifest.entries{comboIndex};
    accuracy = zeros(numel(entries),1);
    for ep = 1:numel(entries)
        episode = raw_episode_from_manifest(raw, features, entries(ep), normalization);
        probability = feat_plain_predict(encoderNet, episode, ...
            temperature, cfg.device);
        accuracy(ep) = local_probability_accuracy(probability, episode);
    end
    combo = manifest.combos(comboIndex,:);
    item = template;
    item.N_WAY = combo(1);
    item.K_SHOT = combo(2);
    item.Q_SHOT = combo(3);
    item.episodes = numel(entries);
    item.acc_mean = mean(accuracy);
    item.acc_std = std(accuracy);
    item.acc_ci95 = 1.96 * item.acc_std / sqrt(numel(accuracy));
    item.episode_accuracy = accuracy;
    result(comboIndex) = item;
end
end

function accuracy = local_probability_accuracy(probability, episode)
[~, localPrediction] = max(gather(extractdata(probability)), [], 1);
prediction = episode.classIds(localPrediction(:));
accuracy = mean(prediction(:) == episode.queryGlobalLabels(:));
end
