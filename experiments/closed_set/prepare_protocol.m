clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();
cfg = benchmark_config("full");
data = load_project_data(cfg, false);

fprintf('[Protocol] Fitting NCA on %d source-train rows.\n', numel(data.fold.source_train));
afopCfg = afop_default_config();
frontend = fit_nca_and_dscan(data.features, data.fold, afopCfg);
fprintf('[Protocol] source-validation D-scan selected D=%d.\n', frontend.selectedD);

evalManifest = generate_episode_manifest(data.features, data.fold.test_all, ...
    cfg.eval.combos, cfg.eval.episodes, cfg.seed.eval, "closedset_test");

stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
artifactPath = fullfile(cfg.paths.artifacts, ['closedset_protocol_' stamp '.mat']);
payload = struct('cfg', cfg, 'frontend', frontend, 'evalManifest', evalManifest);
save_new_artifact(artifactPath, payload);
fprintf('[Protocol] Saved without overwrite: %s\n', artifactPath);
