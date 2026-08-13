clear; clc;
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')));setup_afop_path();
mode=getenv_default('AFOP_MODE','pilot');cfg=benchmark_config(mode);
cfg.paperCWT.imageRoot=getenv_default('AFOP_CWT_IMAGE_ROOT',cfg.paperCWT.imageRoot);
comboIndex=str2double(getenv_default('AFOP_COMBO_INDEX','1'));
data=load_project_data(cfg,false);
protocolPath=find_latest_protocol_artifact(cfg.paths.artifacts);
protocol=load(protocolPath,'evalManifest');
assert(ismember(comboIndex,1:size(cfg.eval.combos,1)),'Invalid combo index.');
combo=cfg.eval.combos(comboIndex,:);
manifest=subset_episode_manifest(protocol.evalManifest,combo,cfg.eval.episodes,data.features);
modelPath=getenv_default('AFOP_CWT_MODEL',latest_cwt_model(cfg.paths.runs,cfg.mode));
saved=load(modelPath,'featureExtractor');rows=data.fold.test_all(:);
embeddings=extract_cwt_embeddings(saved.featureExtractor, ...
    cfg.paperCWT.imageRoot,data.features,rows,cfg.paperCWT,cfg.device);
runInfo=create_run_directory(cfg,"cwt_resnet50",sprintf('%dw%ds_eval',combo(1),combo(2)));
result=evaluate_cwt_manifest(embeddings,rows,data.features,manifest,cfg, ...
    fullfile(runInfo.dir,'checkpoint.mat'),fullfile(runInfo.dir,'progress.log'),comboIndex);
save_new_artifact(fullfile(runInfo.dir,['episodic_' char(cfg.mode) '.mat']), ...
    struct('family',"cwt_resnet50",'modelPath',modelPath,'protocolPath',protocolPath, ...
    'manifest',manifest,'result',result,'cfg',cfg,'runInfo',runInfo));

function path=latest_cwt_model(runsRoot,mode)
files=dir(fullfile(runsRoot,'cwt_resnet50_source_train_*','model.mat'));
assert(~isempty(files),'No CWT-ResNet source checkpoint.');
[~,order]=sort([files.datenum],'descend');path='';fallback='';
for i=order
    candidate=fullfile(files(i).folder,files(i).name);saved=load(candidate,'cfg');
    if string(saved.cfg.mode)=="full",path=candidate;break;end
    if strlength(fallback)==0 && string(saved.cfg.mode)==string(mode),fallback=candidate;end
end
if strlength(path)==0,path=fallback;end
assert(strlength(path)>0,'No full or mode-matched CWT source checkpoint.');
end
function value=getenv_default(name,fallback)
value=getenv(name);if isempty(value),value=fallback;end
end
