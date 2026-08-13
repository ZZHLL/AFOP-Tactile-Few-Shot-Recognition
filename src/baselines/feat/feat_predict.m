function [probability, adaptationMs, inferenceMs] = feat_predict(encoderNet, adapterNet, episode, hp, cfg)
% Inductive FEAT inference using support only for task-specific prototypes.

support = feat_dlarray(episode.supportX, cfg.device);
start = tic;
supportEmbedding = predict(encoderNet, support, 'Outputs','embedding');
supportEmbedding = reshape(stripdims(supportEmbedding), ...
    cfg.featOfficial.embeddingDim, []);
nWay = numel(episode.classIds);
prototypeParts = cell(1,nWay);
for ci = 1:nWay
    prototypeParts{ci} = mean(supportEmbedding(:,episode.localSupport == ci),2);
end
adaptedPrototypes = feat_adapt_set(adapterNet, ...
    cat(2,prototypeParts{:}), false);
adaptationMs = toc(start) * 1000;

query = feat_dlarray(episode.queryX, cfg.device);
start = tic;
queryEmbedding = predict(encoderNet, query, 'Outputs','embedding');
queryEmbedding = reshape(stripdims(queryEmbedding), ...
    cfg.featOfficial.embeddingDim, []);
logits = feat_euclidean_logits(queryEmbedding, ...
    adaptedPrototypes, hp.temperature);
probability = softmax(logits, 'DataFormat','CB');
inferenceMs = toc(start) * 1000;
end
