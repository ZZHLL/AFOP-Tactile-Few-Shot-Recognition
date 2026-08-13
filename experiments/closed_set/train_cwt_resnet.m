clear; clc;
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')));setup_afop_path();
cfg=benchmark_config(getenv_default('AFOP_MODE','full'));
cfg.paperCWT.imageRoot=getenv_default('AFOP_CWT_IMAGE_ROOT',cfg.paperCWT.imageRoot);
data=load_project_data(cfg,false);
assert(isfolder(cfg.paperCWT.imageRoot),'Generate the CWT images first.');
runInfo=create_run_directory(cfg,"cwt_resnet50","source_train");
[featureExtractor,trainInfo]=fit_cwt_resnet(cfg.paperCWT.imageRoot, ...
    data.features,data.fold,cfg.paperCWT,cfg.device,cfg.seed.train+9000000);
outputPath=fullfile(runInfo.dir,'model.mat');
save_new_artifact(outputPath,struct('featureExtractor',featureExtractor, ...
    'trainInfo',trainInfo,'cfg',cfg,'runInfo',runInfo));
fprintf('[CWT-ResNet] Saved: %s\n',outputPath);

function value=getenv_default(name,fallback)
value=getenv(name);if isempty(value),value=fallback;end
end
