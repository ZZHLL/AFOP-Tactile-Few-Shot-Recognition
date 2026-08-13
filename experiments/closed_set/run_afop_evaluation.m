clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();
cfg = benchmark_config(getenv_default('AFOP_MODE', 'full'));
data = load_project_data(cfg, false);
runInfo = create_run_directory(cfg, "afop_reference", "eval");

protocolPath = find_latest_protocol_artifact(cfg.paths.artifacts);
protocol = load(protocolPath, 'frontend','evalManifest');
manifest = subset_episode_manifest(protocol.evalManifest, cfg.eval.combos, ...
    cfg.eval.episodes, data.features);
testRows = data.fold.test_all(:);
columns = protocol.frontend.selectedFeatures(:)';
embeddings = single(data.features.X_all(testRows, columns));
headHP = struct('epochs',250,'learningRate',0.0015,'entropyWeight',0.10,'scale',1);

fprintf('[AFOP reference] D=%d | %d shared episodes per combo.\n', ...
    numel(columns), cfg.eval.episodes);
result = evaluate_embeddings_on_manifest(embeddings, testRows, data.features, manifest, headHP);
for i = 1:numel(result)
    fprintf(['[AFOP reference] %d-way %d-shot: %.2f%% +/- %.2f | ' ...
        'adapt %.1f ms | infer %.4f ms/sample\n'], result(i).N_WAY, ...
        result(i).K_SHOT, 100*result(i).acc_mean, 100*result(i).acc_ci95, ...
        result(i).adapt_ms_median, result(i).infer_ms_per_sample_median);
end
payload = struct('protocolPath',protocolPath,'manifest',manifest,'columns',columns, ...
    'headHP',headHP,'testRows',testRows,'result',result,'cfg',cfg,'runInfo',runInfo);
outputPath = fullfile(runInfo.dir, ['episodic_' char(cfg.mode) '.mat']);
save_new_artifact(outputPath, payload);
fprintf('[AFOP reference] Saved: %s\n', outputPath);

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end
