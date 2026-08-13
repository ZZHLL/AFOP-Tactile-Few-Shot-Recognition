clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root,'src'));
setup_afop_path();
cfg = benchmark_config(getenv_default('AFOP_MODE','pilot'));
data = load_project_data(cfg,true);
protocolPath = find_latest_protocol_artifact(cfg.paths.artifacts);
protocol = load(protocolPath,'evalManifest');
manifest = subset_episode_manifest(protocol.evalManifest,cfg.eval.combos, ...
    cfg.eval.episodes,data.features);
assert(all(ismember(manifest.allowedIdx(:),data.fold.test_all(:))), ...
    'Official FEAT evaluation contains rows outside test_all.');
runInfo = create_run_directory(cfg,"feat_official_raw","eval");

template = struct('N_WAY',[],'K_SHOT',[],'Q_SHOT',[],'episodes',[], ...
    'acc_mean',[],'acc_std',[],'acc_ci95',[],'adapt_ms_median',[], ...
    'infer_ms_per_sample_median',[],'episode_accuracy',[]);
result = repmat(template,size(manifest.combos,1),1);
modelPaths = strings(numel(cfg.featOfficial.shots),1);
for shotIndex = 1:numel(cfg.featOfficial.shots)
    shot = cfg.featOfficial.shots(shotIndex);
    comboRows = find(manifest.combos(:,2) == shot);
    if isempty(comboRows),continue;end
    modelPaths(shotIndex) = find_latest_shot_model(cfg.paths.runs,shot,cfg.mode);
    saved = load(modelPaths(shotIndex),'encoderNet','adapterNet', ...
        'normalization','hp','cfg');
    shotManifest = manifest;
    shotManifest.combos = manifest.combos(comboRows,:);
    shotManifest.entries = manifest.entries(comboRows);
    shotManifest.numEpisodes = cfg.eval.episodes;
    shotResult = evaluate_feat_manifest(saved.encoderNet, ...
        saved.adapterNet,data.raw,data.features,shotManifest, ...
        saved.normalization,saved.hp,saved.cfg);
    result(comboRows) = shotResult;
end

for i = 1:numel(result)
    fprintf(['[Official FEAT] %d-way %d-shot: %.2f%% +/- %.2f | ' ...
        'adapt %.1f ms | infer %.4f ms/sample\n'],result(i).N_WAY, ...
        result(i).K_SHOT,100*result(i).acc_mean,100*result(i).acc_ci95, ...
        result(i).adapt_ms_median,result(i).infer_ms_per_sample_median);
end
payload = struct('modelPaths',modelPaths,'protocolPath',protocolPath, ...
    'manifest',manifest,'result',result,'cfg',cfg,'runInfo',runInfo, ...
    'trainingPolicy',"official-style shot-specific raw-signal FEAT");
outputPath = fullfile(runInfo.dir,['episodic_' char(cfg.mode) '.mat']);
save_new_artifact(outputPath,payload);
fprintf('[Official FEAT] Saved: %s\n',outputPath);

function value = getenv_default(name,fallback)
value = getenv(name);
if isempty(value),value = fallback;end
end

function path = find_latest_shot_model(runsRoot,shot,mode)
family = "feat_official_raw_s" + string(shot) + "_meta";
files = dir(fullfile(runsRoot,family + "_*",'model.mat'));
assert(~isempty(files),'No completed official FEAT s%d model.',shot);
[~,order] = sort([files.datenum],'descend');
path = "";
for index = order
    candidate = fullfile(files(index).folder,files(index).name);
    saved = load(candidate,'cfg');
    if string(saved.cfg.mode) == string(mode)
        path = string(candidate); break;
    end
end
assert(strlength(path)>0,'No official FEAT s%d model for mode %s.',shot,mode);
path = char(path);
end
