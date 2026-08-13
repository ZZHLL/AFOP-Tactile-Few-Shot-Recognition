function [lossValue, accuracy, gradientsEncoder, gradientW, gradientB, encoderState] = ...
        feat_pretrain_gradients(encoderNet, headW, headB, input, labels, weightDecay)
% Compute gradients for supervised encoder pretraining.

[embedding, encoderState] = forward(encoderNet, input, 'Outputs','embedding');
embedding = reshape(stripdims(embedding), size(headW,2), []);
logits = headW * embedding + headB;
[classificationLoss, probability] = feat_classification_loss( ...
    logits, labels, size(headW,1));

l2 = sum(headW.^2, 'all') + sum(headB.^2, 'all');
for i = 1:height(encoderNet.Learnables)
    l2 = l2 + sum(encoderNet.Learnables.Value{i}.^2, 'all');
end
loss = classificationLoss + weightDecay * l2;
[gradientsEncoder, gradientW, gradientB] = dlgradient(loss, ...
    encoderNet.Learnables, headW, headB);
lossValue = double(gather(extractdata(classificationLoss)));
[~, prediction] = max(gather(extractdata(probability)), [], 1);
accuracy = mean(prediction(:) == labels(:));
end
