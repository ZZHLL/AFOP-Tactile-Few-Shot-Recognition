clear; clc;
% Run feature-ranking, fixed-D, and component ablations.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')),'-begin');
addpath(fullfile(root,'configs'),'-begin');
addpath(root,'-begin');
mode = lower(string(getenv_default('AFOP_MODE','full')));
cfg = benchmark_config(mode);
data = load_project_data(cfg,false);
fold = data.fold;

combos = [5 1 15;10 1 15;12 1 15;5 3 15;10 3 15;12 3 15;36 1 15];
episodes = cfg.eval.episodes;
manifest = generate_episode_manifest(data.features,fold.test_all,combos, ...
    episodes,cfg.seed.eval,"feature_optimization_ablation_test");
methods = ["nca","relieff","mrmr","pca"];
fixedD = [4 8 16 32];
headHP = struct('epochs',250,'learningRate',1.5e-3, ...
    'entropyWeight',0.10,'scale',1);

rankingResults = struct([]);
frontendArtifacts = struct();
for methodIndex = 1:numel(methods)
    method = methods(methodIndex);
    frontend = fit_feature_frontend(method,data.features,fold.source_train);
    representation = apply_feature_frontend(frontend,data.features.X_all);
    selection = scan_representation_dimension(representation,data.features, ...
        fold.source_val,cfg.dscan.candidates,cfg);
    selected = representation(:,1:selection.selectedD);
    result = evaluate_embeddings_on_manifest(single(selected(fold.test_all,:)), ...
        fold.test_all,data.features,manifest,headHP);
    rankingResults(methodIndex).method = method;
    rankingResults(methodIndex).selectedD = selection.selectedD;
    rankingResults(methodIndex).selectionRule = selection.rule;
    rankingResults(methodIndex).result = result;
    frontendArtifacts.(char(method)) = struct('frontend',frontend,'selection',selection);
end

nca = frontendArtifacts.nca;
ncaRepresentation = apply_feature_frontend(nca.frontend,data.features.X_all);
fixedDResults = struct([]);
for index = 1:numel(fixedD)
    d = fixedD(index);
    result = evaluate_embeddings_on_manifest(single(ncaRepresentation(fold.test_all,1:d)), ...
        fold.test_all,data.features,manifest,headHP);
    fixedDResults(index) = struct('D',d,'result',result);
end
adaptiveD = nca.selection.selectedD;
fullPool = data.features.X_all;
topD = ncaRepresentation(:,1:adaptiveD);
componentResults = struct();
componentResults.fullPoolPrototype = evaluate_prototype_initialization( ...
    fullPool(fold.test_all,:),fold.test_all,data.features,manifest);
componentResults.topDPrototype = evaluate_prototype_initialization( ...
    topD(fold.test_all,:),fold.test_all,data.features,manifest);
componentResults.fullAFOP = evaluate_embeddings_on_manifest( ...
    single(topD(fold.test_all,:)),fold.test_all,data.features,manifest,headHP);

outputRoot = fullfile(root,'artifacts','ablations');
if ~isfolder(outputRoot),mkdir(outputRoot);end
stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
outputPath = fullfile(outputRoot,['feature_optimization_' stamp '.mat']);
save(outputPath,'cfg','fold','manifest','rankingResults','fixedDResults', ...
    'componentResults','frontendArtifacts','-v7.3');
fprintf('[Ablations] Saved: %s\n',outputPath);
