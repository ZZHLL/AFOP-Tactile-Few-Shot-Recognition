function result = evaluate_embeddings_on_manifest(embeddings, globalRows, features, manifest, headHP)
% Evaluate fixed embeddings on shared episodes with an AFOP-style head.

if ~isfield(headHP,'epochs'), headHP.epochs = 250; end
if ~isfield(headHP,'learningRate'), headHP.learningRate = 0.0015; end
if ~isfield(headHP,'entropyWeight'), headHP.entropyWeight = 0.10; end
if ~isfield(headHP,'scale'), headHP.scale = 1; end

globalRows = globalRows(:);
rowToLocal = zeros(max(globalRows),1);
rowToLocal(globalRows) = 1:numel(globalRows);
template = struct('N_WAY',[],'K_SHOT',[],'Q_SHOT',[],'episodes',[], ...
    'acc_mean',[],'acc_std',[],'acc_ci95',[],'adapt_ms_median',[], ...
    'infer_ms_per_sample_median',[],'episode_accuracy',[]);
allResults = repmat(template, size(manifest.combos,1), 1);

for comboIdx = 1:size(manifest.combos,1)
    combo = manifest.combos(comboIdx,:);
    entries = manifest.entries{comboIdx};
    acc = zeros(numel(entries),1);
    adaptationMs = zeros(numel(entries),1);
    inferenceMs = zeros(numel(entries),1);
    queryCount = zeros(numel(entries),1);
    for ep = 1:numel(entries)
        entry = entries(ep);
        supportRows = reshape(entry.supportIdx', [], 1);
        queryRows = reshape(entry.queryIdx', [], 1);
        supportX = embeddings(rowToLocal(supportRows),:);
        queryX = embeddings(rowToLocal(queryRows),:);
        supportY = features.y_class(supportRows);
        queryY = features.y_class(queryRows);
        [supportX, mu, sigma] = zscore(supportX);
        sigma(sigma == 0) = 1;
        queryX = (queryX - mu) ./ sigma;
        output = fit_episode_cosine_head(supportX, supportY, queryX, entry.classIds, headHP);
        acc(ep) = mean(output.predictions == queryY);
        adaptationMs(ep) = output.adaptationMs;
        inferenceMs(ep) = output.inferenceMs;
        queryCount(ep) = numel(queryY);
    end
    item = struct();
    item.N_WAY = combo(1);
    item.K_SHOT = combo(2);
    item.Q_SHOT = combo(3);
    item.episodes = numel(entries);
    item.acc_mean = mean(acc);
    item.acc_std = std(acc);
    item.acc_ci95 = 1.96*std(acc)/sqrt(numel(acc));
    item.adapt_ms_median = median(adaptationMs);
    item.infer_ms_per_sample_median = median(inferenceMs ./ queryCount);
    item.episode_accuracy = acc;
    allResults(comboIdx) = item;
end
result = allResults;
end

function output = fit_episode_cosine_head(supportX, supportY, queryX, classIds, hp)
n = numel(classIds);
d = size(supportX,2);
weights0 = zeros(n,d,'single');
for ci = 1:n
    w = mean(supportX(supportY == classIds(ci),:),1);
    weights0(ci,:) = w / (norm(w) + 1e-9);
end
weights = dlarray(weights0);
bias = dlarray(zeros(n,1,'single'));
labels = categorical(supportY, classIds);
targets = onehotencode(labels',1);
dlSupport = dlarray(single(supportX'), 'CB');
avgW = []; avgSqW = []; avgB = []; avgSqB = [];

start = tic;
for epoch = 1:hp.epochs
    [gradW, gradB] = dlfeval(@head_gradients, dlSupport, targets, weights, bias, ...
        hp.entropyWeight, hp.scale);
    [weights, avgW, avgSqW] = adamupdate(weights, gradW, avgW, avgSqW, epoch, hp.learningRate);
    [bias, avgB, avgSqB] = adamupdate(bias, gradB, avgB, avgSqB, epoch, hp.learningRate);
end
output.adaptationMs = toc(start)*1000;

start = tic;
probability = cosine_softmax(dlarray(single(queryX'),'CB'), weights, bias, hp.scale);
[~, predictedIndex] = max(gather(extractdata(probability)), [], 1);
predictions = classIds(predictedIndex);
output.inferenceMs = toc(start)*1000;
output.predictions = predictions(:);
end

function [gradW, gradB] = head_gradients(X, targets, weights, bias, entropyWeight, scale)
probability = cosine_softmax(X, weights, bias, scale);
crossEntropy = crossentropy(probability, targets, 'DataFormat','CB');
entropy = -mean(sum(probability .* log(probability + 1e-9), 1));
loss = crossEntropy + entropyWeight*entropy;
[gradW, gradB] = dlgradient(loss, weights, bias);
end

function probability = cosine_softmax(X, weights, bias, scale)
normalWeights = weights ./ (vecnorm(weights,2,2) + 1e-6);
normalX = X ./ (vecnorm(X,2,1) + 1e-6);
scores = stripdims(normalWeights) * stripdims(normalX);
probability = softmax(scale*scores + bias, 'DataFormat','CB');
end
