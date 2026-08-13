clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();
cfg = benchmark_config(getenv_default('AFOP_MODE','smoke'));
shot = str2double(getenv_default('AFOP_FEAT_SHOT','1'));
hp = feat_shot_hyperparameters(cfg,shot);
rng(cfg.seed.train + 20 + shot);
stream = RandStream('mt19937ar','Seed',cfg.seed.train + 20 + shot);
data = load_project_data(cfg,true);
pretrainPath = find_latest_model(cfg.paths.runs,"feat_official_raw","pretrain",cfg.mode);
pretrained = load(pretrainPath,'encoderNet','normalization');
encoderNet = pretrained.encoderNet;
adapterNet = build_feat_adapter(cfg);
runInfo = create_run_directory(cfg,"feat_official_raw_s" + string(shot),"meta");

validationQuery = min(5,6-shot);
validationManifest = generate_episode_manifest(data.features,data.fold.source_val, ...
    [cfg.featOfficial.metaWay shot validationQuery], ...
    cfg.featOfficial.metaValidationEpisodes,cfg.seed.train + 2020 + shot, ...
    "feat_official_meta_validation_s" + string(shot));
assert(isempty(intersect(validationManifest.allowedIdx(:),data.fold.test_all(:))), ...
    'Official FEAT meta-validation leaked into test_all.');

avgEncoder = []; avgSqEncoder = [];
avgAdapter = []; avgSqAdapter = [];
iteration = 0;
encoderLR = cfg.featOfficial.metaLR;
adapterLR = encoderLR * cfg.featOfficial.adapterLRMultiplier;
numValidation = ceil(cfg.featOfficial.metaEpochs / ...
    cfg.featOfficial.metaValidationEvery);
history = zeros(numValidation,7);
historyRow = 0;
bestAccuracy = -inf;
bestEpoch = 0;
bestEncoderNet = encoderNet;
bestAdapterNet = adapterNet;
bestCheckpoint = "";

fprintf(['[Official FEAT meta] shot=%d | epochs=%d | episodes/epoch=%d | ' ...
    'balance=%.2g | T=%.1f | T2=%.1f | mode=%s\n'],shot, ...
    cfg.featOfficial.metaEpochs,cfg.featOfficial.episodesPerEpoch, ...
    hp.balance,hp.temperature,hp.temperature2,cfg.mode);
for epoch = 1:cfg.featOfficial.metaEpochs
    if epoch > 1 && mod(epoch-1,cfg.featOfficial.metaStepEpochs) == 0
        encoderLR = encoderLR * cfg.featOfficial.metaGamma;
        adapterLR = adapterLR * cfg.featOfficial.metaGamma;
    end
    epochLoss = zeros(cfg.featOfficial.episodesPerEpoch,3);
    for ep = 1:cfg.featOfficial.episodesPerEpoch
        episode = sample_raw_episode(data.raw,data.features,data.fold.source_train, ...
            pretrained.normalization,cfg.featOfficial.metaWay,shot, ...
            cfg.featOfficial.metaQuery,stream);
        iteration = iteration + 1;
        [epochLoss(ep,1),epochLoss(ep,2),epochLoss(ep,3),gradientsEncoder, ...
            gradientsAdapter,encoderState] = dlfeval( ...
            @feat_episode_gradients,encoderNet,adapterNet,episode,hp,cfg);
        encoderNet.State = encoderState;
        [encoderNet,avgEncoder,avgSqEncoder] = adamupdate(encoderNet, ...
            gradientsEncoder,avgEncoder,avgSqEncoder,iteration,encoderLR);
        [adapterNet,avgAdapter,avgSqAdapter] = adamupdate(adapterNet, ...
            gradientsAdapter,avgAdapter,avgSqAdapter,iteration,adapterLR);
    end
    if mod(epoch,cfg.featOfficial.metaValidationEvery) == 0 || ...
            epoch == cfg.featOfficial.metaEpochs
        validation = evaluate_feat_manifest(encoderNet,adapterNet, ...
            data.raw,data.features,validationManifest,pretrained.normalization,hp,cfg);
        validationAccuracy = validation(1).acc_mean;
        historyRow = historyRow + 1;
        history(historyRow,:) = [epoch,mean(epochLoss,1),validationAccuracy, ...
            validation(1).acc_ci95,encoderLR];
        fprintf(['[Official FEAT meta] s%d epoch=%d | total=%.4f main=%.4f ' ...
            'aux=%.4f | source-val=%.2f%% +/- %.2f | lr=%.2g/%.2g\n'], ...
            shot,epoch,mean(epochLoss(:,1)),mean(epochLoss(:,2)), ...
            mean(epochLoss(:,3)),100*validationAccuracy, ...
            100*validation(1).acc_ci95,encoderLR,adapterLR);
        if validationAccuracy > bestAccuracy
            bestAccuracy = validationAccuracy;
            bestEpoch = epoch;
            bestEncoderNet = encoderNet;
            bestAdapterNet = adapterNet;
            bestCheckpoint = fullfile(runInfo.checkpoints, ...
                sprintf('best_epoch_%04d.mat',epoch));
            save_new_artifact(bestCheckpoint,struct('encoderNet',encoderNet, ...
                'adapterNet',adapterNet,'epoch',epoch,'validation',validation, ...
                'normalization',pretrained.normalization,'hp',hp,'cfg',cfg));
        end
    end
end

history = array2table(history(1:historyRow,:), 'VariableNames', ...
    {'Epoch','TotalLoss','MainLoss','AuxiliaryLoss','ValidationAccuracy', ...
    'ValidationCI95','EncoderLR'});
payload = struct('encoderNet',bestEncoderNet,'adapterNet',bestAdapterNet, ...
    'bestEpoch',bestEpoch,'bestAccuracy',bestAccuracy, ...
    'bestCheckpoint',bestCheckpoint,'history',history, ...
    'normalization',pretrained.normalization,'hp',hp,'cfg',cfg, ...
    'runInfo',runInfo,'fold',data.fold,'pretrainPath',pretrainPath, ...
    'officialCommit',cfg.featOfficial.officialCommit);
save_new_artifact(fullfile(runInfo.dir,'model.mat'),payload);
fprintf('[Official FEAT meta] s%d best epoch=%d | source-val=%.2f%% | %s\n', ...
    shot,bestEpoch,100*bestAccuracy,runInfo.dir);

function value = getenv_default(name,fallback)
value = getenv(name);
if isempty(value),value = fallback;end
end

function path = find_latest_model(runsRoot,family,phase,mode)
files = dir(fullfile(runsRoot,family + "_" + phase + "_*",'model.mat'));
assert(~isempty(files),'No completed %s %s model.',family,phase);
[~,order] = sort([files.datenum],'descend');
path = "";
for index = order
    candidate = fullfile(files(index).folder,files(index).name);
    saved = load(candidate,'cfg');
    if string(saved.cfg.mode) == string(mode)
        path = string(candidate); break;
    end
end
assert(strlength(path)>0,'No %s %s model for mode %s.',family,phase,mode);
path = char(path);
end
