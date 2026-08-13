function [totalLoss, mainLossValue, auxiliaryLossValue, probability, encoderState] = ...
        feat_episode_objective(encoderNet, adapterNet, episode, hp, cfg)
% FEAT classification objective and class-wise auxiliary regularizer.

allRaw = cat(3, episode.supportX, episode.queryX);
input = feat_dlarray(allRaw, cfg.device);
[allEmbedding, encoderState] = forward(encoderNet, input, 'Outputs','embedding');
allEmbedding = reshape(stripdims(allEmbedding), cfg.featOfficial.embeddingDim, []);
numSupport = size(episode.supportX,3);
supportEmbedding = allEmbedding(:,1:numSupport);
queryEmbedding = allEmbedding(:,numSupport+1:end);
nWay = numel(episode.classIds);

prototypeParts = cell(1,nWay);
for ci = 1:nWay
    prototypeParts{ci} = mean(supportEmbedding(:,episode.localSupport == ci),2);
end
adaptedPrototypes = feat_adapt_set(adapterNet, ...
    cat(2,prototypeParts{:}), true);
mainLogits = feat_euclidean_logits(queryEmbedding, ...
    adaptedPrototypes, hp.temperature);
[mainLoss, probability] = feat_classification_loss( ...
    mainLogits, episode.localQuery, nWay);

auxiliarySamples = cell(1,nWay);
auxiliaryCenters = cell(1,nWay);
auxiliaryLabels = cell(1,nWay);
for ci = 1:nWay
    classSet = [supportEmbedding(:,episode.localSupport == ci), ...
        queryEmbedding(:,episode.localQuery == ci)];
    adaptedClassSet = feat_adapt_set(adapterNet, classSet, true);
    auxiliarySamples{ci} = classSet;
    auxiliaryCenters{ci} = mean(adaptedClassSet,2);
    auxiliaryLabels{ci} = repmat(ci,size(classSet,2),1);
end
auxiliaryLogits = feat_euclidean_logits(cat(2,auxiliarySamples{:}), ...
    cat(2,auxiliaryCenters{:}), hp.temperature2);
[auxiliaryLoss, ~] = feat_classification_loss(auxiliaryLogits, ...
    vertcat(auxiliaryLabels{:}), nWay);
totalLoss = mainLoss + hp.balance * auxiliaryLoss;
mainLossValue = mainLoss;
auxiliaryLossValue = auxiliaryLoss;
end
