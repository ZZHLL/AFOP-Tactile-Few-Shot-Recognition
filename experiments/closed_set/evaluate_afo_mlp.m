clear; clc;
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')));setup_afop_path();
mode=getenv_default('AFOP_MODE','pilot');cfg=benchmark_config(mode);
comboIndex=str2double(getenv_default('AFOP_COMBO_INDEX','1'));
data=load_project_data(cfg,false);
protocolPath=find_latest_protocol_artifact(cfg.paths.artifacts);
protocol=load(protocolPath,'frontend','evalManifest');
assert(ismember(comboIndex,1:size(cfg.eval.combos,1)),'Invalid combo index.');
combo=cfg.eval.combos(comboIndex,:);
manifest=subset_episode_manifest(protocol.evalManifest,combo,cfg.eval.episodes,data.features);
rows=data.fold.test_all(:);columns=protocol.frontend.selectedFeatures(:)';
embeddings=single(data.features.X_all(rows,columns));
runInfo=create_run_directory(cfg,"afo_mlp",sprintf('%dw%ds_eval',combo(1),combo(2)));
result=evaluate_afo_mlp_manifest_checkpointed(embeddings,rows,data.features, ...
    manifest,cfg,fullfile(runInfo.dir,'checkpoint.mat'), ...
    fullfile(runInfo.dir,'progress.log'),comboIndex);
save_new_artifact(fullfile(runInfo.dir,['episodic_' char(cfg.mode) '.mat']), ...
    struct('family',"afo_mlp",'protocolPath',protocolPath,'manifest',manifest, ...
    'columns',columns,'result',result,'cfg',cfg,'runInfo',runInfo));

function value=getenv_default(name,fallback)
value=getenv(name);if isempty(value),value=fallback;end
end
