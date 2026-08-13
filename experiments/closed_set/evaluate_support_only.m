clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();

mode = getenv_default('AFOP_MODE', 'smoke');
family = string(getenv_default('AFOP_MODEL_FAMILY', 'tactile_transformer'));
cfg = benchmark_config(mode);
assert(any(family == ["tactile_transformer","channel_gat"]), 'Unsupported family.');
data = load_project_data(cfg, true);
runInfo = create_run_directory(cfg, family, "support_only_eval");

protocolPath = find_latest_protocol_artifact(cfg.paths.artifacts);
protocol = load(protocolPath, 'evalManifest');
manifest = subset_episode_manifest(protocol.evalManifest, cfg.eval.combos, ...
    cfg.eval.episodes, data.features);

fprintf('[%s support-only] Fresh model per episode; source rows are forbidden.\n', family);
result = evaluate_support_only_raw_model(family, data.raw, data.features, ...
    data.fold, manifest, cfg);
for i = 1:numel(result)
    fprintf('[%s support-only] %d-way %d-shot: %.2f%% +/- %.2f | train %.1f ms | infer %.4f ms/sample\n', ...
        family, result(i).N_WAY, result(i).K_SHOT, 100*result(i).acc_mean, ...
        100*result(i).acc_ci95, result(i).adapt_ms_median, ...
        result(i).infer_ms_per_sample_median);
end

payload = struct('family',family,'mode',cfg.mode,'protocolPath',protocolPath, ...
    'manifest',manifest,'result',result,'cfg',cfg,'runInfo',runInfo, ...
    'trainingPolicy',"fresh support-only model per test episode");
outputPath = fullfile(runInfo.dir, ['episodic_' char(cfg.mode) '.mat']);
save_new_artifact(outputPath, payload);
fprintf('[%s support-only] Saved: %s\n', family, outputPath);

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end
