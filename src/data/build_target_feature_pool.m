function targetFeatures = build_target_feature_pool(rawTarget, labelTemplate)
% Re-extract the 386-D pool from a perturbed target-domain raw dataset.

[numClasses, numTrials] = size(rawTarget);
assert(numTrials == 60 && numClasses >= 2, ...
    'Target raw data must contain 60 trials per class.');
num = struct('shape', numClasses, 'trial', numTrials, 'etr', 1, 'group', 1);
[featurePool, ~] = build_feature_pool(rawTarget, 1000, num);
assert(isequal(size(featurePool), [numClasses*numTrials 386]), ...
    'Unexpected target feature-pool size.');

if istable(labelTemplate)
    targetFeatures = attach_feature_labels(featurePool, labelTemplate);
else
    targetFeatures = labelTemplate;
    targetFeatures.X_all = featurePool;
end
targetFeatures.valid = ~cellfun(@isempty, rawTarget(:));
targetFeatures.bad_mask = ~targetFeatures.valid | ...
    any(~isfinite(targetFeatures.X_all), 2);
assert(~any(targetFeatures.bad_mask), ...
    'The target feature pool contains invalid rows.');
end
