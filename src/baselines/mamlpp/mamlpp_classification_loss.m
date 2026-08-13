function [loss, probability] = mamlpp_classification_loss(encoderParams, headW, headB, input, localLabels, scale)
% Cosine-softmax loss for a task-local classifier.

embedding = mamlpp_encoder(encoderParams, input, scale.strides);
embedding = embedding ./ (sqrt(sum(embedding.^2,1)) + 1e-6);
normalHead = headW ./ (sqrt(sum(headW.^2,2)) + 1e-6);
scores = scale.value * (normalHead * embedding) + headB;
probability = softmax(scores, 'DataFormat','CB');
nWay = size(headW,1);
targets = zeros(nWay, numel(localLabels), 'single');
targets(sub2ind(size(targets), localLabels', 1:numel(localLabels))) = 1;
if isa(extractdata(probability), 'gpuArray')
    targets = gpuArray(targets);
end
loss = -mean(sum(targets .* log(probability + 1e-8), 1));
end
