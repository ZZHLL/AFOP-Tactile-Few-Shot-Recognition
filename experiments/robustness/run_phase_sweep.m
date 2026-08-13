clear; clc;

codeWorkspaceRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
workspaceRoot = getenv_default('AFOP_WORKSPACE_ROOT', codeWorkspaceRoot);
workspaceRoot = char(java.io.File(workspaceRoot).getCanonicalPath());
commonRoot = codeWorkspaceRoot;
addpath(fullfile(commonRoot, 'src'));
addpath(fullfile(commonRoot, 'configs'));
setup_afop_path();

mode = lower(string(getenv_default('AFOP_MODE', 'full')));
assert(any(mode == ["pilot", "full"]), ...
    'Phase sweep supports AFOP_MODE=pilot or full.');
% Pilot mode changes the episode count without changing model hyperparameters.
cfg = benchmark_config("full");
cfg.root = workspaceRoot;
cfg.paths.artifacts = fullfile(workspaceRoot, 'artifacts');
cfg.paths.runs = fullfile(workspaceRoot, 'runs');
cfg.paths.logs = fullfile(workspaceRoot, 'logs');
cfg.input.features = getenv_default('AFOP_FEATURES', ...
    fullfile(commonRoot, 'data', 'features_with_labels.mat'));

shiftSigma = parse_scalar_env('AFOP_SHIFT_SIGMA', 200);
sigmaTag = sprintf('sigma%03d', shiftSigma);
cfg.input.target = getenv_default('AFOP_PHASE_TARGET', ...
    fullfile(cfg.paths.artifacts, ...
    sprintf('phase_target_sigma%03d.mat', shiftSigma)));
referenceProtocolPath = getenv_default('AFOP_REFERENCE_PROTOCOL', ...
    fullfile(workspaceRoot, 'artifacts', 'physical_sigma200_protocol.mat'));

cfg.eval.combos = [5 1 15; 7 1 15; 10 1 15; 12 1 15];
if mode == "pilot"
    cfg.eval.episodes = 50;
elseif mode == "full"
    cfg.eval.episodes = 500;
end
ensure_dir(cfg.paths.artifacts);
ensure_dir(cfg.paths.runs);
ensure_dir(cfg.paths.logs);

assert(isfile(cfg.input.target), 'Missing phase target: %s', cfg.input.target);
assert(isfile(referenceProtocolPath), ...
    'Missing formal sigma-200 protocol: %s', referenceProtocolPath);
featureFile = load(cfg.input.features, 'features_with_labels');
features = featureFile.features_with_labels;
target = load(cfg.input.target, 'dataset_aug', 'info', 'metadata');
validate_target(target.dataset_aug);
assert(target.metadata.shiftSigmaSamples == shiftSigma, ...
    'Target sigma metadata mismatch.');
assert(target.metadata.shiftClipSamples == 3 * shiftSigma, ...
    'Target clip must be 3*sigma.');

reference = load(referenceProtocolPath, 'evalManifest', 'protocol');
manifest = reference.evalManifest;
assert(isequal(manifest.combos, cfg.eval.combos), ...
    'Reference manifest combinations differ from the phase protocol.');
assert(manifest.seed == cfg.seed.eval + 50000, ...
    'Reference manifest seed mismatch.');
assert(all(cellfun(@numel, manifest.entries) == 500), ...
    'Reference manifest must contain 500 episodes per combination.');
if mode == "pilot"
    for comboIndex = 1:numel(manifest.entries)
        manifest.entries{comboIndex} = manifest.entries{comboIndex}(1:50);
    end
end
validate_episode_manifest(manifest, features);

protocol = struct('combos', cfg.eval.combos, ...
    'episodes', cfg.eval.episodes, 'sigmaSamples', shiftSigma, ...
    'clipSamples', 3 * shiftSigma, 'targetMetadata', target.metadata, ...
    'targetInfo', target.info, ...
    'referenceProtocolPath', string(referenceProtocolPath), ...
    'referenceManifestSeed', manifest.seed, ...
    'trainingSource', ...
    "frozen nominal closed-set FEAT and MAML++ models", ...
    'supportOnlyPolicy', ...
    "fresh target-support-only Transformer and Channel-GAT per episode", ...
    'pairingPolicy', ...
    "same support/query episodes, model seeds, amplitude/speed draws across sigma");
if mode == "pilot"
    runTag = "pilot_fullhp";
else
    runTag = "full";
end
protocolPath = fullfile(cfg.paths.artifacts, ...
    sprintf('phase_protocol_%s_%s.mat', sigmaTag, runTag));
if isfile(protocolPath)
    old = load(protocolPath, 'manifest', 'protocol');
    assert(isequal(old.manifest, manifest), ...
        'Existing phase manifest differs.');
    assert(old.protocol.sigmaSamples == shiftSigma, ...
        'Existing phase protocol sigma differs.');
else
    save(protocolPath, 'manifest', 'protocol', '-v7.3');
end

action = lower(string(getenv_default('AFOP_STAGE', 'prepare')));
comboIndex = parse_scalar_env('AFOP_COMBO_INDEX', 1);
runRoot = fullfile(cfg.paths.runs, char(runTag), sigmaTag);
raw = target.dataset_aug;

switch action
    case "prepare"
        fprintf('[phase prepare] verified %s | %s | %d episodes\n', ...
            sigmaTag, protocolPath, cfg.eval.episodes);
    case "afop_eval"
        output = fullfile(runRoot, 'afop', 'episodic_full.mat');
        if isfile(output), fprintf('[phase AFOP] exists: %s\n', output); return; end
        source = load(cfg.input.features, 'features_with_labels');
        targetFeatures = build_target_feature_pool(raw, source.features_with_labels);
        sourceFold = make_closedset_splits(source.features_with_labels, ...
            cfg.fold.trainRatio, cfg.fold.innerTrainRatio, cfg.seed.fold);
        result = evaluate_afop_target_domain(source.features_with_labels, ...
            targetFeatures, sourceFold.source_train, sourceFold.source_val, ...
            manifest, cfg.seed.train + 70000 + shiftSigma, []);
        save_result(output, "AFOP", "", manifest, result, protocol, ...
            "source-domain NCA and D-scan; shifted target queries are evaluation-only");
        print_results('AFOP', result, shiftSigma);
    case "feat_eval"
        output = fullfile(runRoot, 'feat', 'episodic_full.mat');
        if isfile(output), fprintf('[phase FEAT] exists: %s\n', output); return; end
        modelPath = find_latest_shot_model(fullfile(workspaceRoot, 'runs'), 1, "full");
        model = load(modelPath, 'encoderNet', 'adapterNet', ...
            'normalization', 'hp', 'cfg');
        result = evaluate_feat_manifest(model.encoderNet, ...
            model.adapterNet, raw, features, manifest, model.normalization, ...
            model.hp, model.cfg);
        save_result(output, "FEAT", modelPath, manifest, result, protocol, ...
            "frozen nominal-source raw FEAT; no target-query fitting");
        print_results('FEAT', result, shiftSigma);
    case "mamlpp_eval"
        output = fullfile(runRoot, 'mamlpp', 'episodic_full.mat');
        if isfile(output), fprintf('[phase MAML++] exists: %s\n', output); return; end
        modelPath = find_latest_meta_model(fullfile(workspaceRoot, 'runs'), ...
            "mamlpp_raw", "full");
        model = load(modelPath, 'params', 'normalization', 'cfg');
        params = move_mamlpp_parameters(model.params, cfg.device);
        result = evaluate_mamlpp_manifest(params, raw, features, manifest, ...
            model.normalization, model.cfg);
        save_result(output, "MAML++", modelPath, manifest, result, protocol, ...
            "nominal-source raw MAML++; target support-only adaptation");
        print_results('MAML++', result, shiftSigma);
    case {"transformer_eval", "gat_eval"}
        assert(ismember(comboIndex, 1:size(cfg.eval.combos, 1)), ...
            'Invalid combo index.');
        if action == "transformer_eval"
            family = "tactile_transformer";
        else
            family = "channel_gat";
        end
        combo = manifest.combos(comboIndex, :);
        runDir = fullfile(runRoot, char(family), ...
            sprintf('%dw%ds', combo(1), combo(2)));
        output = fullfile(runDir, 'episodic_full.mat');
        if isfile(output), fprintf('[phase %s] exists: %s\n', family, output); return; end
        ensure_dir(runDir);
        one = manifest;
        one.combos = manifest.combos(comboIndex, :);
        one.entries = manifest.entries(comboIndex);
        fold = struct('train_all', zeros(0, 1), ...
            'test_all', (1:numel(features.y_class))');
        result = evaluate_support_only_raw_model_checkpointed(family, raw, ...
            features, fold, one, cfg, fullfile(runDir, 'checkpoint.mat'), ...
            fullfile(runDir, 'progress.log'), 2000 + comboIndex);
        save_result(output, family, "", one, result, protocol, ...
            "fresh support-only raw model per episode; no target-query fitting");
        print_results(char(family), result, shiftSigma);
    otherwise
        error('Unknown AFOP_STAGE: %s', action);
end

function validate_target(raw)
assert(isequal(size(raw), [36 60]), ...
    'Phase target must be 36 x 60 cells.');
for classId = 1:36
    for trialId = 1:60
        X = raw{classId, trialId};
        assert(isequal(size(X), [4 4000]) && all(isfinite(X(:))), ...
            'Invalid target trial at class %d trial %d.', classId, trialId);
    end
end
end

function save_result(path, family, modelPath, manifest, result, protocol, policy)
ensure_dir(fileparts(path));
save_new_artifact(path, struct('family', family, 'modelPath', modelPath, ...
    'manifest', manifest, 'result', result, 'protocol', protocol, ...
    'trainingPolicy', policy));
end

function path = find_latest_shot_model(runsRoot, shot, mode)
files = dir(fullfile(runsRoot, ...
    "feat_official_raw_s" + shot + "_meta_*", 'model.mat'));
assert(~isempty(files), 'No completed FEAT s%d model.', shot);
[~, order] = sort([files.datenum], 'descend');
path = "";
for index = order
    candidate = fullfile(files(index).folder, files(index).name);
    saved = load(candidate, 'cfg');
    if string(saved.cfg.mode) == string(mode)
        path = string(candidate);
        break;
    end
end
assert(strlength(path) > 0, 'No FEAT model for mode %s.', mode);
path = char(path);
end

function path = find_latest_meta_model(runsRoot, family, mode)
files = dir(fullfile(runsRoot, family + "_meta_*", 'model.mat'));
assert(~isempty(files), 'No completed %s model.', family);
[~, order] = sort([files.datenum], 'descend');
path = "";
for index = order
    candidate = fullfile(files(index).folder, files(index).name);
    saved = load(candidate, 'cfg');
    if string(saved.cfg.mode) == string(mode)
        path = string(candidate);
        break;
    end
end
assert(strlength(path) > 0, 'No %s model for mode %s.', family, mode);
path = char(path);
end

function print_results(label, result, shiftSigma)
for index = 1:numel(result)
    fprintf('[phase %d %s] %dw%ds %.2f%% +/- %.2f\n', ...
        shiftSigma, label, result(index).N_WAY, result(index).K_SHOT, ...
        100 * result(index).acc_mean, 100 * result(index).acc_ci95);
end
end

function ensure_dir(path)
if ~exist(path, 'dir'), mkdir(path); end
end

function value = parse_scalar_env(name, fallback)
textValue = getenv(name);
if isempty(textValue)
    value = fallback;
else
    value = str2double(textValue);
    assert(isfinite(value) && isscalar(value), ...
        'Environment variable %s must be a finite scalar.', name);
end
end

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end
