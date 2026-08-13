function selection = scan_representation_dimension(representation,features, ...
    validationRows,candidates,cfg)
% Select D on shared source-validation episodes.

candidates = candidates(candidates<=size(representation,2));
assert(~isempty(candidates),'No valid D candidates.');
combo = [cfg.dscan.way cfg.dscan.shot cfg.dscan.query];
manifest = generate_episode_manifest(features,validationRows,combo, ...
    cfg.dscan.episodes,cfg.seed.dscan,"ablation_dscan_source_val");
curve = zeros(numel(candidates),4);
for index = 1:numel(candidates)
    d = candidates(index);
    accuracy = evaluate_prototype_embeddings(representation(:,1:d), ...
        features,manifest);
    curve(index,:) = [d mean(accuracy) std(accuracy) ...
        1.96*std(accuracy)/sqrt(numel(accuracy))];
end
[~,bestIndex] = max(curve(:,2));
selectedD = curve(bestIndex,1);
rule = "maximum-validation-mean";
selection = struct('selectedD',selectedD,'rule',rule,'manifest',manifest, ...
    'curve',array2table(curve,'VariableNames', ...
    {'D','MeanAccuracy','StdAccuracy','CI95'}));
end

function accuracy = evaluate_prototype_embeddings(X,features,manifest)
entries = manifest.entries{1};
accuracy = zeros(numel(entries),1);
for episode = 1:numel(entries)
    entry = entries(episode);
    supportRows = reshape(entry.supportIdx',[],1);
    queryRows = reshape(entry.queryIdx',[],1);
    supportX = X(supportRows,:);
    queryX = X(queryRows,:);
    supportY = features.y_class(supportRows);
    queryY = features.y_class(queryRows);
    [supportX,mu,sigma] = zscore(supportX); sigma(sigma==0)=1;
    queryX = (queryX-mu)./sigma;
    supportX = normalize(supportX,2); queryX = normalize(queryX,2);
    prototypes = zeros(numel(entry.classIds),size(supportX,2));
    for classIndex = 1:numel(entry.classIds)
        p = mean(supportX(supportY==entry.classIds(classIndex),:),1);
        prototypes(classIndex,:) = p/(norm(p)+1e-9);
    end
    [~,predictionIndex] = max(queryX*prototypes',[],2);
    predictions = entry.classIds(predictionIndex(:));
    accuracy(episode) = mean(predictions(:)==queryY(:));
end
end
