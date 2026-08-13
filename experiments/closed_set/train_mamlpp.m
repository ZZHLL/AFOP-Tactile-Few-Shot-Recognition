clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();
cfg = benchmark_config(getenv_default('AFOP_MODE', 'smoke'));
rng(cfg.seed.train + 3);
stream = RandStream('mt19937ar', 'Seed',cfg.seed.train + 3);
data = load_project_data(cfg, true);
runInfo = create_run_directory(cfg, "mamlpp_raw", "meta");
normalization = fit_raw_channel_normalization(data.raw, data.features, data.fold.source_train);
assert(isequal(sort(normalization.fitIndices(:)), sort(data.fold.source_train(:))), ...
    'MAML++ normalization must be fitted on source_train only.');

protocolPath = find_latest_protocol_artifact(cfg.paths.artifacts);
protocol = load(protocolPath, 'frontend');
validationManifest = subset_episode_manifest(protocol.frontend.validationManifest, ...
    protocol.frontend.validationManifest.combos, cfg.mamlpp.validationEpisodes, data.features);
assert(all(ismember(validationManifest.allowedIdx(:), data.fold.source_val(:))), ...
    'MAML++ checkpoint selection must use source_val only.');
assert(isempty(intersect(validationManifest.allowedIdx(:), data.fold.test_all(:))), ...
    'MAML++ validation leaked into test_all.');

% Arrange each trial as a 4-by-T single-channel input.
params = initialize_mamlpp_parameters(1, cfg, cfg.device);
trailingAverage = [];
trailingAverageSq = [];
numValidations = ceil(cfg.mamlpp.iterations/cfg.mamlpp.validationEvery);
history = zeros(numValidations, 6);
historyRow = 0;
bestAccuracy = -inf;
bestParams = gather_mamlpp_parameters(params);
bestIteration = 0;
bestCheckpoint = "";

fprintf('[MAML++ raw] input=4x%d | iterations=%d | meta-batch=%d | mode=%s\n', ...
    cfg.data.signalLength, cfg.mamlpp.iterations, cfg.mamlpp.metaBatch, cfg.mode);
for iteration = 1:cfg.mamlpp.iterations
    enableHigher = iteration > ceil(cfg.mamlpp.firstOrderFraction*cfg.mamlpp.iterations);
    taskGradients = cell(cfg.mamlpp.metaBatch,1);
    taskLoss = zeros(cfg.mamlpp.metaBatch,1);
    for task = 1:cfg.mamlpp.metaBatch
        episode = sample_raw_episode(data.raw, data.features, data.fold.source_train, ...
            normalization, cfg.mamlpp.metaWay, cfg.mamlpp.metaShot, ...
            cfg.mamlpp.metaQuery, stream);
        [taskLoss(task), taskGradients{task}] = mamlpp_meta_gradients( ...
            params, episode, cfg, enableHigher);
    end
    gradients = average_mamlpp_gradients(taskGradients);
    outerLR = cfg.mamlpp.outerLR * 0.5 * ...
        (1 + cos(pi*(iteration-1)/max(1,cfg.mamlpp.iterations-1)));
    [params, trailingAverage, trailingAverageSq] = adamupdate_mamlpp( ...
        params, gradients, trailingAverage, trailingAverageSq, iteration, outerLR);

    if mod(iteration, cfg.mamlpp.validationEvery) == 0 || iteration == cfg.mamlpp.iterations
        validation = evaluate_mamlpp_manifest(params, data.raw, data.features, ...
            validationManifest, normalization, cfg);
        validationAccuracy = validation(1).acc_mean;
        learnedLR = gather(extractdata(log(1 + exp(params.logInnerLR))));
        historyRow = historyRow + 1;
        history(historyRow,:) = [iteration, mean(taskLoss), validationAccuracy, ...
            validation(1).acc_ci95, outerLR, mean(learnedLR)];
        fprintf(['[MAML++-style] iter=%d | %s | train loss=%.4f | ' ...
            'source-val=%.2f%% +/- %.2f | innerLR=%.5f\n'], ...
            iteration, derivative_label(enableHigher), mean(taskLoss), ...
            100*validationAccuracy, 100*validation(1).acc_ci95, mean(learnedLR));
        if validationAccuracy > bestAccuracy
            bestAccuracy = validationAccuracy;
            bestIteration = iteration;
            bestParams = gather_mamlpp_parameters(params);
            bestCheckpoint = fullfile(runInfo.checkpoints, ...
                sprintf('best_iter_%05d.mat', iteration));
            checkpointPayload = struct('params',bestParams,'iteration',iteration, ...
                'validation',validation,'normalization',normalization,'cfg',cfg);
            save_new_artifact(bestCheckpoint, checkpointPayload);
        end
    end
end
history = array2table(history(1:historyRow,:), 'VariableNames', ...
    {'Iteration','TrainLoss','ValidationAccuracy','ValidationCI95','OuterLR','MeanInnerLR'});
payload = struct('params',bestParams,'bestIteration',bestIteration, ...
    'bestAccuracy',bestAccuracy,'bestCheckpoint',bestCheckpoint,'history',history, ...
    'normalization',normalization,'cfg',cfg,'runInfo',runInfo,'fold',data.fold, ...
    'protocolPath',protocolPath, ...
    'implementationLabel',"MAML++-style raw-signal CNN (adapted)");
save_new_artifact(fullfile(runInfo.dir,'model.mat'), payload);
fprintf('[MAML++ raw] best iter=%d | source-val=%.2f%% | %s\n', ...
    bestIteration, 100*bestAccuracy, runInfo.dir);

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end

function label = derivative_label(enableHigher)
if enableHigher, label = 'second-order'; else, label = 'first-order'; end
end
