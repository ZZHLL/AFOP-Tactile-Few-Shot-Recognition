clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();
cfg = benchmark_config(getenv_default('AFOP_MODE', 'smoke'));
rng(cfg.seed.train + 10);
stream = RandStream('mt19937ar', 'Seed',cfg.seed.train + 10);
data = load_project_data(cfg, true);
runInfo = create_run_directory(cfg, "feat_official_raw", "pretrain");
normalization = fit_raw_channel_normalization(data.raw, data.features, data.fold.source_train);
assert(isempty(intersect(normalization.fitIndices(:), data.fold.test_all(:))), ...
    'FEAT pretraining normalization leaked into test_all.');

validationManifest = generate_episode_manifest(data.features, data.fold.source_val, ...
    [5 1 5], cfg.featOfficial.pretrainValidationEpisodes, ...
    cfg.seed.train + 1010, "feat_official_pretrain_validation");
encoderNet = build_feat_encoder(cfg);
[headW, headB] = initialize_head(cfg.data.numClasses, ...
    cfg.featOfficial.embeddingDim, cfg.device);
avgEncoder = []; avgSqEncoder = [];
avgW = []; avgSqW = []; avgB = []; avgSqB = [];
iteration = 0;
currentLR = cfg.featOfficial.pretrainLR;
bestAccuracy = -inf;
bestEpoch = 0;
bestEncoderNet = encoderNet;
bestCheckpoint = "";
history = zeros(cfg.featOfficial.pretrainEpochs, 5);
historyRow = 0;
sourceRows = data.fold.source_train(:);

fprintf('[Official FEAT pretrain] epochs=%d | batch=%d | mode=%s\n', ...
    cfg.featOfficial.pretrainEpochs, cfg.featOfficial.pretrainBatchSize, cfg.mode);
for epoch = 1:cfg.featOfficial.pretrainEpochs
    if ismember(epoch, cfg.featOfficial.pretrainSchedule)
        currentLR = currentLR * cfg.featOfficial.pretrainGamma;
    end
    order = sourceRows(randperm(stream, numel(sourceRows)));
    batchLoss = zeros(ceil(numel(order)/cfg.featOfficial.pretrainBatchSize),1);
    batchAccuracy = zeros(size(batchLoss));
    batchIndex = 0;
    for startIndex = 1:cfg.featOfficial.pretrainBatchSize:numel(order)
        batchIndex = batchIndex + 1;
        batchRows = order(startIndex:min(startIndex + ...
            cfg.featOfficial.pretrainBatchSize - 1, numel(order)));
        rawBatch = normalize_raw_tensor(raw_batch_from_indices( ...
            data.raw, data.features, batchRows), normalization);
        input = feat_dlarray(rawBatch, cfg.device);
        labels = data.features.y_class(batchRows);
        iteration = iteration + 1;
        [batchLoss(batchIndex), batchAccuracy(batchIndex), gradientsEncoder, ...
            gradientW, gradientB, encoderState] = dlfeval( ...
            @feat_pretrain_gradients, encoderNet, headW, headB, ...
            input, labels, cfg.featOfficial.pretrainWeightDecay);
        encoderNet.State = encoderState;
        [encoderNet, avgEncoder, avgSqEncoder] = adamupdate(encoderNet, ...
            gradientsEncoder, avgEncoder, avgSqEncoder, iteration, currentLR);
        [headW, avgW, avgSqW] = adamupdate(headW, gradientW, ...
            avgW, avgSqW, iteration, currentLR);
        [headB, avgB, avgSqB] = adamupdate(headB, gradientB, ...
            avgB, avgSqB, iteration, currentLR);
    end

    shouldValidate = mod(epoch-1, cfg.featOfficial.pretrainValidationEvery) == 0 || ...
        epoch > cfg.featOfficial.pretrainDenseValidationAfter || ...
        epoch == cfg.featOfficial.pretrainEpochs;
    if shouldValidate
        validation = evaluate_feat_plain_manifest(encoderNet, data.raw, ...
            data.features, validationManifest, normalization, 64, cfg);
        historyRow = historyRow + 1;
        history(historyRow,:) = [epoch,mean(batchLoss),mean(batchAccuracy), ...
            validation.acc_mean,validation.acc_ci95];
        fprintf(['[Official FEAT pretrain] epoch=%d | loss=%.4f | train=%.2f%% | ' ...
            'source-val proto=%.2f%% +/- %.2f | lr=%.2g\n'], epoch, ...
            mean(batchLoss),100*mean(batchAccuracy),100*validation.acc_mean, ...
            100*validation.acc_ci95,currentLR);
        if validation.acc_mean > bestAccuracy
            bestAccuracy = validation.acc_mean;
            bestEpoch = epoch;
            bestEncoderNet = encoderNet;
            bestCheckpoint = fullfile(runInfo.checkpoints, ...
                sprintf('best_epoch_%04d.mat',epoch));
            save_new_artifact(bestCheckpoint, struct('encoderNet',encoderNet, ...
                'epoch',epoch,'validation',validation,'normalization',normalization, ...
                'cfg',cfg));
        end
    end
end

history = array2table(history(1:historyRow,:), 'VariableNames', ...
    {'Epoch','TrainLoss','TrainAccuracy','ValidationAccuracy','ValidationCI95'});
payload = struct('encoderNet',bestEncoderNet,'bestEpoch',bestEpoch, ...
    'bestAccuracy',bestAccuracy,'bestCheckpoint',bestCheckpoint,'history',history, ...
    'normalization',normalization,'cfg',cfg,'runInfo',runInfo,'fold',data.fold, ...
    'officialCommit',cfg.featOfficial.officialCommit);
save_new_artifact(fullfile(runInfo.dir,'model.mat'), payload);
fprintf('[Official FEAT pretrain] best epoch=%d | source-val=%.2f%% | %s\n', ...
    bestEpoch,100*bestAccuracy,runInfo.dir);

function [W, B] = initialize_head(numClasses, embeddingDim, device)
limit = sqrt(6 / (numClasses + embeddingDim));
W = dlarray(single((2*rand(numClasses,embeddingDim)-1)*limit));
B = dlarray(zeros(numClasses,1,'single'));
if device == "gpu"
    W = dlarray(gpuArray(extractdata(W)));
    B = dlarray(gpuArray(extractdata(B)));
end
end

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end
