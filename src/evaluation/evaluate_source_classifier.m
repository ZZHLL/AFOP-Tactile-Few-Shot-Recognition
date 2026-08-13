function metrics = evaluate_source_classifier(net, sequences, labels, device, miniBatchSize)
% Independently compute source-domain loss and accuracy from classifier logits.

if nargin < 5 || isempty(miniBatchSize)
    miniBatchSize = 32;
end
labels = double(labels(:));
numSamples = numel(sequences);
assert(numel(labels) == numSamples, 'Sequence and label counts must match.');

totalLoss = 0;
numCorrect = 0;
allFinite = true;
for first = 1:miniBatchSize:numSamples
    last = min(first + miniBatchSize - 1, numSamples);
    batch = cat(3, sequences{first:last});
    batch = permute(batch, [2 1 3]);
    if device == "gpu"
        batch = gpuArray(batch);
    end
    logits = predict(net, dlarray(batch, 'CTB'), 'Outputs','classifier');
    logits = double(gather(extractdata(logits)));
    allFinite = allFinite && all(isfinite(logits), 'all');

    logits = logits - max(logits, [], 1);
    logProbabilities = logits - log(sum(exp(logits), 1));
    batchLabels = labels(first:last)';
    linear = sub2ind(size(logProbabilities), batchLabels, 1:numel(batchLabels));
    totalLoss = totalLoss - sum(logProbabilities(linear));
    [~, predicted] = max(logits, [], 1);
    numCorrect = numCorrect + sum(predicted == batchLabels);
end

metrics.loss = totalLoss / numSamples;
metrics.accuracy = numCorrect / numSamples;
metrics.numSamples = numSamples;
metrics.allFinite = allFinite;
end
