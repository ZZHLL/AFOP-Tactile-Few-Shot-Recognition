function probability = feat_plain_predict(encoderNet, episode, temperature, device)
% ProtoNet-style validation of a pretrained FEAT encoder.

support = feat_dlarray(episode.supportX, device);
query = feat_dlarray(episode.queryX, device);
supportEmbedding = predict(encoderNet, support, 'Outputs','embedding');
queryEmbedding = predict(encoderNet, query, 'Outputs','embedding');
supportEmbedding = reshape(stripdims(supportEmbedding), [], size(episode.supportX,3));
queryEmbedding = reshape(stripdims(queryEmbedding), [], size(episode.queryX,3));
nWay = numel(episode.classIds);
prototypes = cell(1,nWay);
for ci = 1:nWay
    prototypes{ci} = mean(supportEmbedding(:, episode.localSupport == ci), 2);
end
logits = feat_euclidean_logits(queryEmbedding, cat(2,prototypes{:}), temperature);
probability = softmax(logits, 'DataFormat','CB');
end
