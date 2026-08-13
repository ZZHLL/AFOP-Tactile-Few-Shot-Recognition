clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();
cfg = benchmark_config(getenv_default('AFOP_MODE', 'pilot'));
data = load_project_data(cfg, true);

modelPath = find_latest_meta_model(cfg.paths.runs, "mamlpp_raw", cfg.mode);
protocolPath = find_latest_protocol_artifact(cfg.paths.artifacts);
saved = load(modelPath, 'params','normalization','cfg','runInfo');
params = move_mamlpp_parameters(saved.params, cfg.device);
protocol = load(protocolPath, 'evalManifest');
manifest = subset_episode_manifest(protocol.evalManifest, cfg.eval.combos, ...
    cfg.eval.episodes, data.features);
assert(all(ismember(manifest.allowedIdx(:), data.fold.test_all(:))), ...
    'MAML++ evaluation manifest contains rows outside test_all.');
assert(isempty(intersect(saved.normalization.fitIndices(:), data.fold.test_all(:))), ...
    'MAML++ normalization contains held-out test rows.');

fprintf('[MAML++ raw] Evaluating %d shared episodes per combo.\n', cfg.eval.episodes);
result = evaluate_mamlpp_manifest(params, data.raw, data.features, manifest, ...
    saved.normalization, saved.cfg);
for i = 1:numel(result)
    fprintf(['[MAML++ raw] %d-way %d-shot: %.2f%% +/- %.2f | ' ...
        'adapt %.1f ms | infer %.4f ms/sample\n'], result(i).N_WAY, ...
        result(i).K_SHOT, 100*result(i).acc_mean, 100*result(i).acc_ci95, ...
        result(i).adapt_ms_median, result(i).infer_ms_per_sample_median);
end
payload = struct('modelPath',modelPath,'protocolPath',protocolPath, ...
    'manifest',manifest,'result',result, ...
    'trainingPolicy',"raw-signal episodic meta-training; support-only test adaptation");
outputPath = fullfile(saved.runInfo.dir, ['episodic_' char(cfg.mode) '.mat']);
save_new_artifact(outputPath, payload);
fprintf('[MAML++ raw] Saved: %s\n', outputPath);

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end

function modelPath = find_latest_meta_model(runsRoot, family, mode)
matches = dir(fullfile(runsRoot, family + "_meta_*", 'model.mat'));
assert(~isempty(matches), 'No completed model found for %s.', family);
[~, order] = sort([matches.datenum], 'descend');
modelPath = "";
for index = order
    candidate = fullfile(matches(index).folder, matches(index).name);
    candidateCfg = load(candidate, 'cfg');
    if string(candidateCfg.cfg.mode) == string(mode)
        modelPath = string(candidate);
        break;
    end
end
assert(strlength(modelPath) > 0, 'No %s model found for mode %s.', family, mode);
modelPath = char(modelPath);
end
