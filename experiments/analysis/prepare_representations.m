clear; clc;
% Prepare AFOP, parent-pool, and modality-specific embeddings for CKA/t-SNE.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')),'-begin');
addpath(fullfile(root,'configs'),'-begin');
addpath(root,'-begin');
cfg = benchmark_config(lower(string(getenv_default('AFOP_MODE','full'))));
data = load_project_data(cfg,false);
fold = data.fold;

definitions = struct( ...
    'AFOP',1:size(data.features.X_all,2), ...
    'SGOnly',361:386, ...
    'PVDFOnly',1:360);
names = string(fieldnames(definitions));
representations = struct();
selection = struct();
for index = 1:numel(names)
    name = names(index);
    columns = definitions.(name);
    masked = data.features;
    masked.X_all = data.features.X_all(:,columns);
    frontend = fit_feature_frontend("nca",masked,fold.source_train);
    ranked = apply_feature_frontend(frontend,masked.X_all);
    selected = scan_representation_dimension(ranked,masked,fold.source_val, ...
        cfg.dscan.candidates,cfg);
    representations.(name) = ranked(fold.test_all,1:selected.selectedD);
    selection.(name) = struct('sourceColumns',columns, ...
        'rankWithinMask',frontend.rank,'selectedD',selected.selectedD, ...
        'curve',selected.curve,'rule',selected.rule);
end
representations.DirectProto = data.features.X_all(fold.test_all,:);
rows = fold.test_all(:);
labels = struct('class',data.features.y_class(rows), ...
    'shape',data.features.y_shape12(rows), ...
    'material',data.features.y_material(rows), ...
    'trial',data.features.y_trial(rows));
outputRoot = fullfile(root,'artifacts','representation_analysis');
if ~isfolder(outputRoot),mkdir(outputRoot);end
stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
outputPath = fullfile(outputRoot,['feature_representations_' stamp '.mat']);
save(outputPath,'representations','selection','rows','labels','fold','cfg','-v7.3');
fprintf('[Representation input] Saved: %s\n',outputPath);
