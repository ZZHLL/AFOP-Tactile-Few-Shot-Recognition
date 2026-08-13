function [loss, probability] = feat_classification_loss(logits, localLabels, nWay)
% Cross-entropy over local episode labels.

probability = softmax(logits, 'DataFormat','CB');
targets = zeros(nWay, numel(localLabels), 'single');
targets(sub2ind(size(targets), localLabels', 1:numel(localLabels))) = 1;
if isa(extractdata(probability), 'gpuArray')
    targets = gpuArray(targets);
end
loss = -mean(sum(targets .* log(probability + 1e-8), 1));
end
