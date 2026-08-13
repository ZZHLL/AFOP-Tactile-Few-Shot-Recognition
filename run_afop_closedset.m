function output = run_afop_closedset(dataDirectory,outputFile)
% Run the minimal closed-set AFOP pipeline from prepared jittered trials.
%
% Required files in dataDirectory:
%   dataset2.mat     variable dataset3_jittered (36-by-60 cells)
%   labels_table.mat variable labels_table

packageDirectory = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(packageDirectory,'src')),'-begin');
addpath(fullfile(packageDirectory,'configs'),'-begin');
addpath(packageDirectory,'-begin');

if nargin<1 || isempty(dataDirectory)
    dataDirectory = fullfile(packageDirectory,'data');
end
if nargin<2 || isempty(outputFile)
    outputFile = fullfile(packageDirectory,'afop_results.mat');
end
datasetFile = fullfile(dataDirectory,'dataset2.mat');
labelsFile = fullfile(dataDirectory,'labels_table.mat');
assert(isfile(datasetFile),'Missing %s',datasetFile);
assert(isfile(labelsFile),'Missing %s',labelsFile);

cfg = afop_default_config();
raw = load(datasetFile,'dataset3_jittered');
labels = load(labelsFile,'labels_table');
dataset = raw.dataset3_jittered;
assert(isequal(size(dataset),[cfg.numClasses cfg.trialsPerClass]), ...
    'Expected a 36-by-60 cell array.');

num = struct('shape',cfg.numClasses,'trial',cfg.trialsPerClass,'etr',1,'group',1);
[featurepool,~,featureMeta] = build_feature_pool(dataset,cfg.fs,num);
features = attach_feature_labels(featurepool,labels.labels_table);
assert(isequal(size(features.X_all),[2160 386]) && ~any(features.bad_mask), ...
    'Invalid feature pool.');

fold = make_closedset_splits(features,cfg.trainRatio, ...
    cfg.innerTrainRatio,cfg.seed.split);
validate_closedset_split(features,fold,cfg);
frontend = fit_nca_and_dscan(features,fold,cfg);
manifest = generate_episode_manifest(features,fold.test_all,cfg.eval.combos, ...
    cfg.eval.episodes,cfg.seed.evalBase);
results = evaluate_afop(features,frontend.selectedFeatures,manifest, ...
    cfg.afop,cfg.seed.evalBase);

output = struct('cfg',cfg,'featureMeta',featureMeta,'fold',fold, ...
    'frontend',frontend,'manifest',manifest,'results',results);
save(outputFile,'-struct','output','-v7.3');
end

function validate_closedset_split(features,fold,cfg)
y = features.y_class(:);
assert(isempty(intersect(fold.train_all,fold.test_all)),'Train/test overlap.');
assert(isempty(intersect(fold.source_train,fold.source_val)),'Train/val overlap.');
for c = 1:cfg.numClasses
    assert(nnz(y(fold.train_all)==c)==30 && nnz(y(fold.test_all)==c)==30, ...
        'Class %d does not have a 30/30 split.',c);
    assert(nnz(y(fold.source_train)==c)==24 && nnz(y(fold.source_val)==c)==6, ...
        'Class %d does not have a 24/6 source split.',c);
end
end
