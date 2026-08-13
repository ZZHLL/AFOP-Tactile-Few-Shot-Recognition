clear; clc;

codeWorkspaceRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
workspaceRoot = getenv_default('AFOP_WORKSPACE_ROOT', codeWorkspaceRoot);
workspaceRoot = char(java.io.File(workspaceRoot).getCanonicalPath());
commonRoot = codeWorkspaceRoot;
addpath(fullfile(commonRoot, 'src'));
addpath(fullfile(commonRoot, 'configs'));
setup_afop_path();

mode = lower(string(getenv_default('AFOP_MODE', 'full')));
cfg = benchmark_config(mode);
cfg.root = workspaceRoot;
cfg.paths.artifacts = fullfile(workspaceRoot, 'artifacts');
cfg.paths.runs = fullfile(workspaceRoot, 'runs');
cfg.paths.logs = fullfile(workspaceRoot, 'logs');
cfg.input.dataset = getenv_default('AFOP_DATASET', ...
    fullfile(commonRoot, 'data', 'dataset2.mat'));
cfg.input.labels = getenv_default('AFOP_LABELS', ...
    fullfile(commonRoot, 'data', 'labels_table.mat'));
cfg.input.features = getenv_default('AFOP_FEATURES', ...
    fullfile(commonRoot, 'data', 'features_with_labels.mat'));
cfg.eval.combos = [5 1 15; 7 1 15; 10 1 15; 12 1 15];
if mode == "smoke"
    cfg.eval.episodes = 3;
elseif mode == "pilot"
    cfg.eval.episodes = 50;
else
    cfg.eval.episodes = 500;
end

ensure_dir(cfg.paths.artifacts);
ensure_dir(cfg.paths.runs);
ensure_dir(cfg.paths.logs);
ensure_dir(fullfile(workspaceRoot, 'state'));

action = lower(string(getenv_default('AFOP_STAGE', 'prepare')));
splitId = str2double(getenv_default('AFOP_SPLIT', '1'));
comboIndex = str2double(getenv_default('AFOP_COMBO_INDEX', '1'));

if action == "prepare"
    prepare_protocols(cfg);
    return;
end

assert(isfinite(splitId) && ismember(splitId, 1:3), 'AFOP_SPLIT must be 1, 2, or 3.');
data = load_base_data(cfg);
protocolFile = fullfile(cfg.paths.artifacts, sprintf('crossmaterial_split%d.mat', splitId));
assert(isfile(protocolFile), 'Missing protocol artifact: %s', protocolFile);
protocol = load(protocolFile, 'fold', 'evalManifest', 'metadata');
data.fold = protocol.fold;

fprintf('\n[Cross-material] split=%d source=%s target=%s | stage=%s\n', ...
    splitId, protocol.metadata.sourceMaterial, ...
    strjoin(protocol.metadata.targetMaterials, '+'), action);

switch action
    case "afop_eval"
        evaluate_afop_domain(data, cfg, splitId, protocol);
    case "feat_pretrain"
        train_feat_pretrain(data, cfg, splitId, protocolFile);
    case "feat_meta"
        train_feat_meta(data, cfg, splitId, protocolFile);
    case "feat_eval"
        evaluate_feat(data, cfg, splitId, protocol);
    case "mamlpp_train"
        train_mamlpp_domain(data, cfg, splitId, protocolFile);
    case "mamlpp_eval"
        evaluate_mamlpp_domain(data, cfg, splitId, protocol);
    case {"transformer_eval", "gat_eval"}
        assert(isfinite(comboIndex) && ismember(comboIndex, 1:size(cfg.eval.combos,1)), ...
            'AFOP_COMBO_INDEX is invalid.');
        evaluate_support_only(action, data, cfg, splitId, protocol, comboIndex);
    otherwise
        error('Unknown AFOP_STAGE: %s', action);
end

function evaluate_afop_domain(data, cfg, splitId, protocol)
output = fullfile(cfg.paths.runs, 'crossmaterial', sprintf('split_%d', splitId), ...
    'afop_eval', 'episodic_full.mat');
if isfile(output), fprintf('[AFOP eval] exists: %s\n', output); return; end
selection = cfg.dscan;
selection.seed = 20250813;
result = evaluate_afop_target_domain(data.features, data.features, ...
    data.fold.source_train, data.fold.source_val, protocol.evalManifest, ...
    cfg.seed.train + 1000*splitId, selection);
ensure_dir(fileparts(output));
save_new_artifact(output, struct('family', "AFOP", ...
    'manifest', protocol.evalManifest, 'result', result, ...
    'trainingPolicy', "NCA and D-scan use the source material only; target query is evaluation-only"));
print_results('AFOP', result);
end

function prepare_protocols(cfg)
data = load_base_data(cfg);
combos = cfg.eval.combos;
materialNames = ["Resin", "Metal", "Wood"];
for splitId = 1:3
    sourceRows = find(data.features.valid(:) & data.features.y_material(:) == splitId);
    targetRows = find(data.features.valid(:) & data.features.y_material(:) ~= splitId);
    assert(numel(sourceRows) == 12*60, 'Unexpected source row count for material %d.', splitId);
    assert(numel(targetRows) == 24*60, 'Unexpected target row count for material %d.', splitId);

    stream = RandStream('mt19937ar', 'Seed', cfg.seed.fold + 100*splitId);
    sourceTrain = zeros(12*48, 1);
    sourceVal = zeros(12*12, 1);
    trainCursor = 0;
    valCursor = 0;
    sourceClasses = unique(data.features.y_class(sourceRows))';
    assert(numel(sourceClasses) == 12, 'A source material must contain 12 shape classes.');
    for classId = sourceClasses
        rows = sourceRows(data.features.y_class(sourceRows) == classId);
        rows = rows(randperm(stream, numel(rows)));
        sourceTrain(trainCursor + (1:48)) = rows(1:48);
        sourceVal(valCursor + (1:12)) = rows(49:60);
        trainCursor = trainCursor + 48;
        valCursor = valCursor + 12;
    end
    fold = struct('train_all', sourceRows(:), 'source_train', sourceTrain(:), ...
        'source_val', sourceVal(:), 'test_all', targetRows(:));
    assert(isempty(intersect(fold.source_train, fold.source_val)), 'Source train/val overlap.');
    assert(isempty(intersect(fold.train_all, fold.test_all)), 'Source/target overlap.');
    assert(isequal(sort([fold.source_train; fold.source_val]), sort(fold.train_all)), ...
        'Source train/val do not cover the source material exactly.');

    evalSeed = cfg.seed.eval + 10000*splitId;
    evalManifest = generate_episode_manifest(data.features, targetRows, combos, ...
        cfg.eval.episodes, evalSeed, "crossmaterial_shared_target_split" + splitId);
    metadata = struct();
    metadata.version = 1;
    metadata.splitId = splitId;
    metadata.sourceMaterialId = splitId;
    metadata.sourceMaterial = materialNames(splitId);
    metadata.targetMaterialIds = setdiff(1:3, splitId);
    metadata.targetMaterials = materialNames(metadata.targetMaterialIds);
    metadata.sourceTrainPerClass = 48;
    metadata.sourceValPerClass = 12;
    metadata.targetTrialsPerClass = 60;
    metadata.combos = combos;
    metadata.episodes = cfg.eval.episodes;
    metadata.evalSeed = evalSeed;
    metadata.rawInput = "dataset3_jittered, 4 x 4000";
    metadata.created = string(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));

    output = fullfile(cfg.paths.artifacts, sprintf('crossmaterial_split%d.mat', splitId));
    if isfile(output)
        old = load(output, 'fold', 'evalManifest', 'metadata');
        assert(isequal(old.fold, fold), 'Existing split artifact differs: %s', output);
        assert(isequal(old.evalManifest, evalManifest), 'Existing manifest differs: %s', output);
        fprintf('[prepare] verified existing %s\n', output);
    else
        save(output, 'fold', 'evalManifest', 'metadata', '-v7.3');
        fprintf('[prepare] saved %s\n', output);
    end
end
end

function data = load_base_data(cfg)
required = {cfg.input.dataset, cfg.input.labels, cfg.input.features};
for i = 1:numel(required)
    assert(isfile(required{i}), 'Missing input: %s', required{i});
end
rawFile = load(cfg.input.dataset, 'dataset3_jittered');
labelsFile = load(cfg.input.labels, 'labels_table');
featureFile = load(cfg.input.features, 'features_with_labels');
data = struct();
data.raw = rawFile.dataset3_jittered;
data.labelsTable = labelsFile.labels_table;
data.features = featureFile.features_with_labels;
assert(isequal(size(data.raw), [36 60]), 'Raw dataset must be 36 x 60 cells.');
assert(numel(data.features.y_class) == 2160, 'Feature metadata must have 2160 rows.');
probe = data.raw{1,1};
assert(isequal(size(probe), [4 4000]), 'Raw trial must be 4 x 4000.');
end

function train_feat_pretrain(data, cfg, splitId, protocolFile)
pointer = stage_pointer(cfg, splitId, 'feat_pretrain');
if isfile(pointer)
    fprintf('[FEAT pretrain] completed pointer exists: %s\n', pointer);
    return;
end
rng(cfg.seed.train + 1000*splitId + 10);
stream = RandStream('mt19937ar', 'Seed', cfg.seed.train + 1000*splitId + 10);
runInfo = create_split_run(cfg, splitId, "feat_official_raw", "pretrain");
normalization = fit_raw_channel_normalization(data.raw, data.features, data.fold.source_train);
assert(isempty(intersect(normalization.fitIndices(:), data.fold.test_all(:))), ...
    'FEAT normalization leaked into the target domain.');
validationManifest = generate_episode_manifest(data.features, data.fold.source_val, ...
    [5 1 5], cfg.featOfficial.pretrainValidationEpisodes, ...
    cfg.seed.train + 1000*splitId + 1010, "crossmaterial_feat_pretrain_validation");
encoderNet = build_feat_encoder(cfg);
[headW, headB] = initialize_head(cfg.data.numClasses, cfg.featOfficial.embeddingDim, cfg.device);
avgEncoder = []; avgSqEncoder = []; avgW = []; avgSqW = []; avgB = []; avgSqB = [];
iteration = 0; currentLR = cfg.featOfficial.pretrainLR; bestAccuracy = -inf;
bestEpoch = 0; bestEncoderNet = encoderNet; bestCheckpoint = "";
history = zeros(cfg.featOfficial.pretrainEpochs, 5); historyRow = 0;
sourceRows = data.fold.source_train(:);
for epoch = 1:cfg.featOfficial.pretrainEpochs
    if ismember(epoch, cfg.featOfficial.pretrainSchedule), currentLR = currentLR*cfg.featOfficial.pretrainGamma; end
    order = sourceRows(randperm(stream, numel(sourceRows)));
    batchLoss = zeros(ceil(numel(order)/cfg.featOfficial.pretrainBatchSize),1);
    batchAccuracy = zeros(size(batchLoss)); batchIndex = 0;
    for startIndex = 1:cfg.featOfficial.pretrainBatchSize:numel(order)
        batchIndex = batchIndex + 1;
        rows = order(startIndex:min(startIndex+cfg.featOfficial.pretrainBatchSize-1,numel(order)));
        rawBatch = normalize_raw_tensor(raw_batch_from_indices(data.raw,data.features,rows),normalization);
        input = feat_dlarray(rawBatch,cfg.device);
        labels = data.features.y_class(rows); iteration = iteration + 1;
        [batchLoss(batchIndex),batchAccuracy(batchIndex),gE,gW,gB,state] = dlfeval( ...
            @feat_pretrain_gradients,encoderNet,headW,headB,input,labels, ...
            cfg.featOfficial.pretrainWeightDecay);
        encoderNet.State = state;
        [encoderNet,avgEncoder,avgSqEncoder] = adamupdate(encoderNet,gE,avgEncoder,avgSqEncoder,iteration,currentLR);
        [headW,avgW,avgSqW] = adamupdate(headW,gW,avgW,avgSqW,iteration,currentLR);
        [headB,avgB,avgSqB] = adamupdate(headB,gB,avgB,avgSqB,iteration,currentLR);
    end
    shouldValidate = mod(epoch-1,cfg.featOfficial.pretrainValidationEvery)==0 || ...
        epoch > cfg.featOfficial.pretrainDenseValidationAfter || epoch==cfg.featOfficial.pretrainEpochs;
    if shouldValidate
        validation = evaluate_feat_plain_manifest(encoderNet,data.raw,data.features, ...
            validationManifest,normalization,64,cfg);
        historyRow = historyRow + 1;
        history(historyRow,:) = [epoch,mean(batchLoss),mean(batchAccuracy),validation.acc_mean,validation.acc_ci95];
        fprintf('[FEAT pretrain s%d] epoch=%d loss=%.4f source-val=%.2f%% +/- %.2f\n', ...
            splitId,epoch,mean(batchLoss),100*validation.acc_mean,100*validation.acc_ci95);
        if validation.acc_mean > bestAccuracy
            bestAccuracy = validation.acc_mean; bestEpoch = epoch; bestEncoderNet = encoderNet;
            bestCheckpoint = fullfile(runInfo.checkpoints,sprintf('best_epoch_%04d.mat',epoch));
            save_new_artifact(bestCheckpoint,struct('encoderNet',encoderNet,'epoch',epoch, ...
                'validation',validation,'normalization',normalization,'cfg',cfg));
        end
    end
end
history = array2table(history(1:historyRow,:), 'VariableNames', ...
    {'Epoch','TrainLoss','TrainAccuracy','ValidationAccuracy','ValidationCI95'});
modelPath = fullfile(runInfo.dir,'model.mat');
save_new_artifact(modelPath,struct('encoderNet',bestEncoderNet,'bestEpoch',bestEpoch, ...
    'bestAccuracy',bestAccuracy,'bestCheckpoint',bestCheckpoint,'history',history, ...
    'normalization',normalization,'cfg',cfg,'runInfo',runInfo,'fold',data.fold, ...
    'protocolFile',protocolFile,'officialCommit',cfg.featOfficial.officialCommit));
save_pointer(pointer,modelPath,runInfo.dir);
end

function train_feat_meta(data, cfg, splitId, protocolFile)
pointer = stage_pointer(cfg, splitId, 'feat_meta');
if isfile(pointer), fprintf('[FEAT meta] completed pointer exists: %s\n',pointer); return; end
pretrain = load_pointer(stage_pointer(cfg,splitId,'feat_pretrain'));
saved = load(pretrain.modelPath,'encoderNet','normalization');
encoderNet = saved.encoderNet; normalization = saved.normalization;
shot = 1; hp = feat_shot_hyperparameters(cfg,shot);
rng(cfg.seed.train + 1000*splitId + 20 + shot);
stream = RandStream('mt19937ar','Seed',cfg.seed.train + 1000*splitId + 20 + shot);
runInfo = create_split_run(cfg,splitId,"feat_official_raw_s1","meta");
adapterNet = build_feat_adapter(cfg);
validationManifest = generate_episode_manifest(data.features,data.fold.source_val, ...
    [cfg.featOfficial.metaWay shot min(5,6-shot)],cfg.featOfficial.metaValidationEpisodes, ...
    cfg.seed.train + 1000*splitId + 2020 + shot,"crossmaterial_feat_meta_validation");
avgEncoder=[];avgSqEncoder=[];avgAdapter=[];avgSqAdapter=[];iteration=0;
encoderLR=cfg.featOfficial.metaLR;adapterLR=encoderLR*cfg.featOfficial.adapterLRMultiplier;
history=zeros(ceil(cfg.featOfficial.metaEpochs/cfg.featOfficial.metaValidationEvery),7);historyRow=0;
bestAccuracy=-inf;bestEpoch=0;bestEncoderNet=encoderNet;bestAdapterNet=adapterNet;bestCheckpoint="";
for epoch=1:cfg.featOfficial.metaEpochs
    if epoch>1 && mod(epoch-1,cfg.featOfficial.metaStepEpochs)==0
        encoderLR=encoderLR*cfg.featOfficial.metaGamma;adapterLR=adapterLR*cfg.featOfficial.metaGamma;
    end
    epochLoss=zeros(cfg.featOfficial.episodesPerEpoch,3);
    for ep=1:cfg.featOfficial.episodesPerEpoch
        episode=sample_raw_episode(data.raw,data.features,data.fold.source_train,normalization, ...
            cfg.featOfficial.metaWay,shot,cfg.featOfficial.metaQuery,stream);
        iteration=iteration+1;
        [epochLoss(ep,1),epochLoss(ep,2),epochLoss(ep,3),gE,gA,state]=dlfeval( ...
            @feat_episode_gradients,encoderNet,adapterNet,episode,hp,cfg);
        encoderNet.State=state;
        [encoderNet,avgEncoder,avgSqEncoder]=adamupdate(encoderNet,gE,avgEncoder,avgSqEncoder,iteration,encoderLR);
        [adapterNet,avgAdapter,avgSqAdapter]=adamupdate(adapterNet,gA,avgAdapter,avgSqAdapter,iteration,adapterLR);
    end
    if mod(epoch,cfg.featOfficial.metaValidationEvery)==0 || epoch==cfg.featOfficial.metaEpochs
        validation=evaluate_feat_manifest(encoderNet,adapterNet,data.raw,data.features, ...
            validationManifest,normalization,hp,cfg);
        va=validation(1).acc_mean;historyRow=historyRow+1;
        history(historyRow,:)=[epoch,mean(epochLoss,1),va,validation(1).acc_ci95,encoderLR];
        fprintf('[FEAT meta split%d] epoch=%d source-val=%.2f%% +/- %.2f\n', ...
            splitId,epoch,100*va,100*validation(1).acc_ci95);
        if va>bestAccuracy
            bestAccuracy=va;bestEpoch=epoch;bestEncoderNet=encoderNet;bestAdapterNet=adapterNet;
            bestCheckpoint=fullfile(runInfo.checkpoints,sprintf('best_epoch_%04d.mat',epoch));
            save_new_artifact(bestCheckpoint,struct('encoderNet',encoderNet,'adapterNet',adapterNet, ...
                'epoch',epoch,'validation',validation,'normalization',normalization,'hp',hp,'cfg',cfg));
        end
    end
end
history=array2table(history(1:historyRow,:),'VariableNames', ...
    {'Epoch','TotalLoss','MainLoss','AuxiliaryLoss','ValidationAccuracy','ValidationCI95','EncoderLR'});
modelPath=fullfile(runInfo.dir,'model.mat');
save_new_artifact(modelPath,struct('encoderNet',bestEncoderNet,'adapterNet',bestAdapterNet, ...
    'bestEpoch',bestEpoch,'bestAccuracy',bestAccuracy,'bestCheckpoint',bestCheckpoint, ...
    'history',history,'normalization',normalization,'hp',hp,'cfg',cfg,'runInfo',runInfo, ...
    'fold',data.fold,'pretrainPath',pretrain.modelPath,'protocolFile',protocolFile, ...
    'officialCommit',cfg.featOfficial.officialCommit));
save_pointer(pointer,modelPath,runInfo.dir);
end

function evaluate_feat(data,cfg,splitId,protocol)
output=fullfile(cfg.paths.runs,'crossmaterial',sprintf('split_%d',splitId),'feat_eval','episodic_full.mat');
if isfile(output),fprintf('[FEAT eval] exists: %s\n',output);return;end
meta=load_pointer(stage_pointer(cfg,splitId,'feat_meta'));
saved=load(meta.modelPath,'encoderNet','adapterNet','normalization','hp','cfg');
result=evaluate_feat_manifest(saved.encoderNet,saved.adapterNet,data.raw,data.features, ...
    protocol.evalManifest,saved.normalization,saved.hp,saved.cfg);
ensure_dir(fileparts(output));
payload=struct('family',"FEAT",'modelPath',meta.modelPath,'manifest',protocol.evalManifest, ...
    'result',result,'trainingPolicy',"source-trained raw FEAT; target support/query evaluation");
save_new_artifact(output,payload);print_results('FEAT',result);
end

function train_mamlpp_domain(data,cfg,splitId,protocolFile)
pointer=stage_pointer(cfg,splitId,'mamlpp_train');
if isfile(pointer),fprintf('[MAML++] completed pointer exists: %s\n',pointer);return;end
rng(cfg.seed.train+1000*splitId+3);
stream=RandStream('mt19937ar','Seed',cfg.seed.train+1000*splitId+3);
runInfo=create_split_run(cfg,splitId,"mamlpp_raw","meta");
normalization=fit_raw_channel_normalization(data.raw,data.features,data.fold.source_train);
validationManifest=generate_episode_manifest(data.features,data.fold.source_val,[5 1 5], ...
    cfg.mamlpp.validationEpisodes,cfg.seed.train+1000*splitId+3030,"crossmaterial_mamlpp_validation");
params=initialize_mamlpp_parameters(1,cfg,cfg.device);trailingAverage=[];trailingAverageSq=[];
history=zeros(ceil(cfg.mamlpp.iterations/cfg.mamlpp.validationEvery),6);historyRow=0;
bestAccuracy=-inf;bestParams=gather_mamlpp_parameters(params);bestIteration=0;bestCheckpoint="";
for iteration=1:cfg.mamlpp.iterations
    enableHigher=iteration>ceil(cfg.mamlpp.firstOrderFraction*cfg.mamlpp.iterations);
    taskGradients=cell(cfg.mamlpp.metaBatch,1);taskLoss=zeros(cfg.mamlpp.metaBatch,1);
    for task=1:cfg.mamlpp.metaBatch
        episode=sample_raw_episode(data.raw,data.features,data.fold.source_train,normalization, ...
            cfg.mamlpp.metaWay,cfg.mamlpp.metaShot,cfg.mamlpp.metaQuery,stream);
        [taskLoss(task),taskGradients{task}]=mamlpp_meta_gradients(params,episode,cfg,enableHigher);
    end
    gradients=average_mamlpp_gradients(taskGradients);
    outerLR=cfg.mamlpp.outerLR*0.5*(1+cos(pi*(iteration-1)/max(1,cfg.mamlpp.iterations-1)));
    [params,trailingAverage,trailingAverageSq]=adamupdate_mamlpp(params,gradients, ...
        trailingAverage,trailingAverageSq,iteration,outerLR);
    if mod(iteration,cfg.mamlpp.validationEvery)==0 || iteration==cfg.mamlpp.iterations
        validation=evaluate_mamlpp_manifest(params,data.raw,data.features,validationManifest,normalization,cfg);
        va=validation(1).acc_mean;learnedLR=gather(extractdata(log(1+exp(params.logInnerLR))));
        historyRow=historyRow+1;history(historyRow,:)=[iteration,mean(taskLoss),va, ...
            validation(1).acc_ci95,outerLR,mean(learnedLR)];
        fprintf('[MAML++ split%d] iter=%d source-val=%.2f%% +/- %.2f\n', ...
            splitId,iteration,100*va,100*validation(1).acc_ci95);
        if va>bestAccuracy
            bestAccuracy=va;bestIteration=iteration;bestParams=gather_mamlpp_parameters(params);
            bestCheckpoint=fullfile(runInfo.checkpoints,sprintf('best_iter_%05d.mat',iteration));
            save_new_artifact(bestCheckpoint,struct('params',bestParams,'iteration',iteration, ...
                'validation',validation,'normalization',normalization,'cfg',cfg));
        end
    end
end
history=array2table(history(1:historyRow,:),'VariableNames', ...
    {'Iteration','TrainLoss','ValidationAccuracy','ValidationCI95','OuterLR','MeanInnerLR'});
modelPath=fullfile(runInfo.dir,'model.mat');
save_new_artifact(modelPath,struct('params',bestParams,'bestIteration',bestIteration, ...
    'bestAccuracy',bestAccuracy,'bestCheckpoint',bestCheckpoint,'history',history, ...
    'normalization',normalization,'cfg',cfg,'runInfo',runInfo,'fold',data.fold, ...
    'protocolFile',protocolFile,'implementationLabel',"MAML++-style raw-signal CNN (adapted)"));
save_pointer(pointer,modelPath,runInfo.dir);
end

function evaluate_mamlpp_domain(data,cfg,splitId,protocol)
output=fullfile(cfg.paths.runs,'crossmaterial',sprintf('split_%d',splitId),'mamlpp_eval','episodic_full.mat');
if isfile(output),fprintf('[MAML++ eval] exists: %s\n',output);return;end
meta=load_pointer(stage_pointer(cfg,splitId,'mamlpp_train'));
saved=load(meta.modelPath,'params','normalization','cfg');
params=move_mamlpp_parameters(saved.params,cfg.device);
result=evaluate_mamlpp_manifest(params,data.raw,data.features,protocol.evalManifest,saved.normalization,saved.cfg);
ensure_dir(fileparts(output));
save_new_artifact(output,struct('family',"MAML++",'modelPath',meta.modelPath, ...
    'manifest',protocol.evalManifest,'result',result, ...
    'trainingPolicy',"source-trained raw MAML++; target support-only adaptation"));
print_results('MAML++',result);
end

function evaluate_support_only(action,data,cfg,splitId,protocol,comboIndex)
if action=="transformer_eval",family="tactile_transformer";else,family="channel_gat";end
combo=protocol.evalManifest.combos(comboIndex,:);
runDir=fullfile(cfg.paths.runs,'crossmaterial',sprintf('split_%d',splitId),char(family), ...
    sprintf('%dw%ds',combo(1),combo(2)));
output=fullfile(runDir,'episodic_full.mat');
if isfile(output),fprintf('[%s] exists: %s\n',family,output);return;end
ensure_dir(runDir);
manifest=protocol.evalManifest;manifest.combos=manifest.combos(comboIndex,:);manifest.entries=manifest.entries(comboIndex);
checkpointPath=fullfile(runDir,'checkpoint.mat');progressPath=fullfile(runDir,'progress.log');
seedIndex=100*splitId+comboIndex;
result=evaluate_support_only_raw_model_checkpointed(family,data.raw,data.features,data.fold, ...
    manifest,cfg,checkpointPath,progressPath,seedIndex);
save_new_artifact(output,struct('family',family,'manifest',manifest,'result',result, ...
    'cfg',cfg,'trainingPolicy',"fresh support-only raw model per target episode"));
print_results(char(family),result);
end

function runInfo=create_split_run(cfg,splitId,modelName,stage)
originalRuns=cfg.paths.runs;
cfg.paths.runs=fullfile(originalRuns,'crossmaterial',sprintf('split_%d',splitId));
ensure_dir(cfg.paths.runs);
runInfo=create_run_directory(cfg,modelName,stage);
end

function path=stage_pointer(cfg,splitId,name)
path=fullfile(cfg.root,'state',sprintf('crossmaterial_split%d_%s.mat',splitId,name));
end

function save_pointer(path,modelPath,runDir)
assert(~isfile(path),'Refusing to overwrite stage pointer: %s',path);
completed=string(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
save(path,'modelPath','runDir','completed','-v7.3');
end

function value=load_pointer(path)
assert(isfile(path),'Required completed stage is missing: %s',path);value=load(path);
end

function [W,B]=initialize_head(numClasses,embeddingDim,device)
limit=sqrt(6/(numClasses+embeddingDim));
W=dlarray(single((2*rand(numClasses,embeddingDim)-1)*limit));B=dlarray(zeros(numClasses,1,'single'));
if device=="gpu",W=dlarray(gpuArray(extractdata(W)));B=dlarray(gpuArray(extractdata(B)));end
end

function print_results(label,result)
for i=1:numel(result)
    fprintf('[%s] %dw%ds %.2f%% +/- %.2f\n',label,result(i).N_WAY,result(i).K_SHOT, ...
        100*result(i).acc_mean,100*result(i).acc_ci95);
end
end

function ensure_dir(path)
if ~exist(path,'dir'),mkdir(path);end
end

function value=getenv_default(name,fallback)
value=getenv(name);if isempty(value),value=fallback;end
end
