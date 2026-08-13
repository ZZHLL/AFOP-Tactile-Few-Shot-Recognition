clear; clc;
% Rebuild the feature pool and evaluate one wavelet setting.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')),'-begin');
addpath(fullfile(root,'configs'),'-begin');
addpath(root,'-begin');

study = lower(string(getenv_default('AFOP_WAVELET_STUDY','kernel')));
settingIndex = str2double(getenv_default('AFOP_WAVELET_INDEX','1'));
kernels = ["db5","coif5","sym5","fk6","dmey","bior2.2", ...
    "rbio2.2","bior2.6","rbio2.6"];
levels = 1:8;
assert(settingIndex==round(settingIndex) && settingIndex>=1, ...
    'AFOP_WAVELET_INDEX must be a positive integer.');

switch study
    case "kernel"
        assert(settingIndex<=numel(kernels),'Kernel index exceeds %d.',numel(kernels));
        featureConfig = struct('wavelet',kernels(settingIndex),'levels',3);
    case "level"
        assert(settingIndex<=numel(levels),'Level index exceeds %d.',numel(levels));
        featureConfig = struct('wavelet',"rbio2.2",'levels',levels(settingIndex));
    otherwise
        error('AFOP_WAVELET_STUDY must be kernel or level.');
end

datasetPath = string(getenv_default('AFOP_DATASET',fullfile(root,'data','dataset2.mat')));
labelsPath = string(getenv_default('AFOP_LABELS',fullfile(root,'data','labels_table.mat')));
assert(isfile(datasetPath),'Missing dataset: %s',datasetPath);
assert(isfile(labelsPath),'Missing labels: %s',labelsPath);
raw = load(datasetPath,'dataset3_jittered');
labelSource = load(labelsPath,'labels_table');

cfg = afop_default_config();
num = struct('shape',cfg.numClasses,'trial',cfg.trialsPerClass,'etr',1,'group',1);
[featurepool,~,featureMeta] = build_feature_pool( ...
    raw.dataset3_jittered,cfg.fs,num,featureConfig);
features = attach_feature_labels(featurepool,labelSource.labels_table);
fold = make_closedset_splits(features,cfg.trainRatio, ...
    cfg.innerTrainRatio,cfg.seed.split);

frontend = fit_nca_and_dscan(features,fold,cfg);
combos = [5 1 15;12 1 15;36 1 15];
manifest = generate_episode_manifest(features,fold.test_all,combos, ...
    500,cfg.seed.evalBase,"wavelet_ablation_test");
results = evaluate_afop(features,frontend.selectedFeatures,manifest, ...
    cfg.afop,cfg.seed.evalBase);

outputRoot = fullfile(root,'artifacts','wavelet_ablation');
if ~isfolder(outputRoot), mkdir(outputRoot); end
safeWavelet = strrep(char(featureConfig.wavelet),'.','_');
stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
outputPath = fullfile(outputRoot,sprintf('%s_%s_L%d_%s.mat', ...
    study,safeWavelet,featureConfig.levels,stamp));
save(outputPath,'study','featureConfig','featureMeta','cfg','fold', ...
    'frontend','manifest','results','-v7.3');
fprintf('[Wavelet ablation] %s L%d | pool=%d | D*=%d | %s\n', ...
    featureConfig.wavelet,featureConfig.levels,size(featurepool,2), ...
    frontend.selectedD,outputPath);
