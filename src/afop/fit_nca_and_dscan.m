function frontend = fit_nca_and_dscan(features, fold, cfg)
% Fit NCA on source_train and select D exclusively on source_val episodes.
% D* is the candidate with the highest mean 5-way 5-shot validation accuracy.

X = features.X_all;
y = features.y_class(:);
trainIdx = fold.source_train(:);
valIdx = fold.source_val(:);

[XTrainScaledT, minmaxSettings] = mapminmax(X(trainIdx,:)', 0, 1);
XTrainScaled = XTrainScaledT';
nca = fscnca(XTrainScaled, y(trainIdx), 'Verbose', 0);
[~, rank] = sort(nca.FeatureWeights(:), 'descend');

combos = [cfg.dscan.way cfg.dscan.shot cfg.dscan.query];
valManifest = generate_episode_manifest(features, valIdx, combos, ...
    cfg.dscan.episodes, cfg.seed.dscan, "dscan_source_val");

curve = zeros(numel(cfg.dscan.candidates), 4);
for di = 1:numel(cfg.dscan.candidates)
    d = cfg.dscan.candidates(di);
    selected = rank(1:d);
    acc = zeros(cfg.dscan.episodes,1);
    for ep = 1:cfg.dscan.episodes
        entry = valManifest.entries{1}(ep);
        [supportX, supportY, queryX, queryY] = feature_episode(features, entry, selected);
        [supportX, mu, sigma] = zscore(supportX);
        sigma(sigma == 0) = 1;
        queryX = (queryX - mu) ./ sigma;
        predictions = nearest_cosine_prototypes(supportX, supportY, queryX, entry.classIds);
        acc(ep) = mean(predictions == queryY);
    end
    curve(di,:) = [d, mean(acc), std(acc), 1.96*std(acc)/sqrt(numel(acc))];
end

[~, bestIdx] = max(curve(:,2));
chosenD = curve(bestIdx,1);

frontend = struct();
frontend.method = "NCA";
frontend.rank = rank;
frontend.weights = nca.FeatureWeights(:);
frontend.minmaxSettings = minmaxSettings;
frontend.selectedD = chosenD;
frontend.selectedFeatures = rank(1:chosenD);
frontend.curve = array2table(curve, 'VariableNames', {'D','MeanAccuracy','StdAccuracy','CI95'});
frontend.validationManifest = valManifest;
frontend.fitRows = trainIdx;
frontend.selectionRows = valIdx;
frontend.selectionRule = "maximum-validation-mean";
end

function [supportX, supportY, queryX, queryY] = feature_episode(features, entry, columns)
supportRows = entry.supportIdx';
queryRows = entry.queryIdx';
supportRows = supportRows(:);
queryRows = queryRows(:);
supportX = features.X_all(supportRows, columns);
queryX = features.X_all(queryRows, columns);
supportY = features.y_class(supportRows);
queryY = features.y_class(queryRows);
end

function predictions = nearest_cosine_prototypes(supportX, supportY, queryX, classIds)
supportX = normalize(supportX, 2);
queryX = normalize(queryX, 2);
prototypes = zeros(numel(classIds), size(supportX,2));
for ci = 1:numel(classIds)
    p = mean(supportX(supportY == classIds(ci),:), 1);
    prototypes(ci,:) = p / (norm(p) + 1e-9);
end
[~, index] = max(queryX * prototypes', [], 2);
predictions = classIds(index(:));
predictions = predictions(:);
end
