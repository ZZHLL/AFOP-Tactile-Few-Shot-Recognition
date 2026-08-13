clear; clc;

codeWorkspaceRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
workspaceRoot = getenv_default('AFOP_WORKSPACE_ROOT', codeWorkspaceRoot);
workspaceRoot = char(java.io.File(workspaceRoot).getCanonicalPath());
commonRoot = codeWorkspaceRoot;
addpath(fullfile(commonRoot,'src'));
addpath(fullfile(commonRoot,'configs'));
setup_afop_path();

mode = lower(string(getenv_default('AFOP_MODE','full')));
cfg = benchmark_config(mode);
cfg.root = workspaceRoot;
cfg.paths.artifacts = fullfile(workspaceRoot,'artifacts');
cfg.paths.runs = fullfile(workspaceRoot,'runs');
cfg.paths.logs = fullfile(workspaceRoot,'logs');
cfg.input.target = getenv_default('AFOP_PHYSICAL_TARGET', ...
    fullfile(cfg.paths.artifacts,'physical_target_sigma200.mat'));
cfg.input.features = getenv_default('AFOP_FEATURES', ...
    fullfile(commonRoot,'data','features_with_labels.mat'));
cfg.eval.combos = [5 1 15;7 1 15;10 1 15;12 1 15];
if mode=="smoke",cfg.eval.episodes=3;elseif mode=="pilot",cfg.eval.episodes=50;else,cfg.eval.episodes=500;end
ensure_dir(cfg.paths.artifacts);ensure_dir(cfg.paths.runs);ensure_dir(cfg.paths.logs);

action = lower(string(getenv_default('AFOP_STAGE','prepare')));
comboIndex = str2double(getenv_default('AFOP_COMBO_INDEX','1'));
featureFile = load(cfg.input.features,'features_with_labels');
features = featureFile.features_with_labels;

if action == "prepare"
    assert(isfile(cfg.input.target),'Missing physical target: %s',cfg.input.target);
    target = load(cfg.input.target,'dataset_aug','info','metadata');
    validate_target(target.dataset_aug);
    allowed = (1:numel(features.y_class))';
    evalManifest = generate_episode_manifest(features,allowed,cfg.eval.combos, ...
        cfg.eval.episodes,cfg.seed.eval+50000,"physical_sigma200_shared_target");
    protocol = struct('combos',cfg.eval.combos,'episodes',cfg.eval.episodes, ...
        'targetMetadata',target.metadata,'targetInfo',target.info, ...
        'trainingSource',"audited nominal closed-set models for FEAT and MAML++", ...
        'supportOnlyPolicy',"fresh target-support-only Transformer and Channel-GAT");
    output=fullfile(cfg.paths.artifacts,'physical_sigma200_protocol.mat');
    if isfile(output)
        old=load(output,'evalManifest','protocol');
        assert(isequal(old.evalManifest,evalManifest),'Existing physical manifest differs.');
        fprintf('[physical prepare] verified %s\n',output);
    else
        save(output,'evalManifest','protocol','-v7.3');
        fprintf('[physical prepare] saved %s\n',output);
    end
    return;
end

target=load(cfg.input.target,'dataset_aug');raw=target.dataset_aug;validate_target(raw);
savedProtocol=load(fullfile(cfg.paths.artifacts,'physical_sigma200_protocol.mat'), ...
    'evalManifest','protocol');
manifest=savedProtocol.evalManifest;

switch action
    case "afop_eval"
        output=fullfile(cfg.paths.runs,'physical_sigma200','afop','episodic_full.mat');
        if isfile(output),fprintf('[physical AFOP] exists: %s\n',output);return;end
        source=load(cfg.input.features,'features_with_labels');
        targetFeatures=build_target_feature_pool(raw,source.features_with_labels);
        sourceFold=make_closedset_splits(source.features_with_labels, ...
            cfg.fold.trainRatio,cfg.fold.innerTrainRatio,cfg.seed.fold);
        result=evaluate_afop_target_domain(source.features_with_labels,targetFeatures, ...
            sourceFold.source_train,sourceFold.source_val,manifest,cfg.seed.train+60000,[]);
        save_result(output,"AFOP","",manifest,result, ...
            "source-domain NCA and D-scan; perturbed target queries are evaluation-only");
        print_results('AFOP',result);
    case "feat_eval"
        output=fullfile(cfg.paths.runs,'physical_sigma200','feat','episodic_full.mat');
        if isfile(output),fprintf('[physical FEAT] exists: %s\n',output);return;end
        modelPath=find_latest_shot_model(fullfile(workspaceRoot,'runs'),1,"full");
        model=load(modelPath,'encoderNet','adapterNet','normalization','hp','cfg');
        result=evaluate_feat_manifest(model.encoderNet,model.adapterNet,raw,features, ...
            manifest,model.normalization,model.hp,model.cfg);
        save_result(output,"FEAT",modelPath,manifest,result, ...
            "nominal-source raw FEAT; frozen before physical-target evaluation");
        print_results('FEAT',result);
    case "mamlpp_eval"
        output=fullfile(cfg.paths.runs,'physical_sigma200','mamlpp','episodic_full.mat');
        if isfile(output),fprintf('[physical MAML++] exists: %s\n',output);return;end
        modelPath=find_latest_meta_model(fullfile(workspaceRoot,'runs'),"mamlpp_raw","full");
        model=load(modelPath,'params','normalization','cfg');
        params=move_mamlpp_parameters(model.params,cfg.device);
        result=evaluate_mamlpp_manifest(params,raw,features,manifest,model.normalization,model.cfg);
        save_result(output,"MAML++",modelPath,manifest,result, ...
            "nominal-source raw MAML++; target support-only adaptation");
        print_results('MAML++',result);
    case {"transformer_eval","gat_eval"}
        assert(ismember(comboIndex,1:size(cfg.eval.combos,1)),'Invalid combo index.');
        if action=="transformer_eval",family="tactile_transformer";else,family="channel_gat";end
        combo=manifest.combos(comboIndex,:);
        runDir=fullfile(cfg.paths.runs,'physical_sigma200',char(family),sprintf('%dw%ds',combo(1),combo(2)));
        output=fullfile(runDir,'episodic_full.mat');
        if isfile(output),fprintf('[physical %s] exists: %s\n',family,output);return;end
        ensure_dir(runDir);
        one=manifest;one.combos=manifest.combos(comboIndex,:);one.entries=manifest.entries(comboIndex);
        fold=struct('train_all',zeros(0,1),'test_all',(1:numel(features.y_class))');
        result=evaluate_support_only_raw_model_checkpointed(family,raw,features,fold,one,cfg, ...
            fullfile(runDir,'checkpoint.mat'),fullfile(runDir,'progress.log'),2000+comboIndex);
        save_result(output,family,"",one,result,"fresh support-only raw model per physical-target episode");
        print_results(char(family),result);
    otherwise
        error('Unknown AFOP_STAGE: %s',action);
end

function validate_target(raw)
assert(isequal(size(raw),[36 60]),'Physical target must be 36 x 60 cells.');
for classId=[1 12 24 36]
    for trialId=[1 30 60]
        X=raw{classId,trialId};
        assert(isequal(size(X),[4 4000]) && all(isfinite(X(:))),'Invalid target trial.');
    end
end
end

function save_result(path,family,modelPath,manifest,result,policy)
ensure_dir(fileparts(path));
save_new_artifact(path,struct('family',family,'modelPath',modelPath,'manifest',manifest, ...
    'result',result,'trainingPolicy',policy));
end

function path=find_latest_shot_model(runsRoot,shot,mode)
files=dir(fullfile(runsRoot,"feat_official_raw_s"+shot+"_meta_*",'model.mat'));
assert(~isempty(files),'No completed FEAT s%d model.',shot);[~,order]=sort([files.datenum],'descend');path="";
for index=order
    candidate=fullfile(files(index).folder,files(index).name);saved=load(candidate,'cfg');
    if string(saved.cfg.mode)==string(mode),path=string(candidate);break;end
end
assert(strlength(path)>0,'No FEAT model for mode %s.',mode);path=char(path);
end

function path=find_latest_meta_model(runsRoot,family,mode)
files=dir(fullfile(runsRoot,family+"_meta_*",'model.mat'));
assert(~isempty(files),'No completed %s model.',family);[~,order]=sort([files.datenum],'descend');path="";
for index=order
    candidate=fullfile(files(index).folder,files(index).name);saved=load(candidate,'cfg');
    if string(saved.cfg.mode)==string(mode),path=string(candidate);break;end
end
assert(strlength(path)>0,'No %s model for mode %s.',family,mode);path=char(path);
end

function print_results(label,result)
for i=1:numel(result)
    fprintf('[physical %s] %dw%ds %.2f%% +/- %.2f\n',label,result(i).N_WAY, ...
        result(i).K_SHOT,100*result(i).acc_mean,100*result(i).acc_ci95);
end
end

function ensure_dir(path)
if ~exist(path,'dir'),mkdir(path);end
end

function value=getenv_default(name,fallback)
value=getenv(name);if isempty(value),value=fallback;end
end
