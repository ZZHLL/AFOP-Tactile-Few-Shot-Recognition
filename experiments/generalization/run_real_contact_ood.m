clear; clc;
% Real-contact OOD evaluation on 18 target classes (6 shapes x 3 materials).

root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')));setup_afop_path();
mode=lower(string(getenv_default('AFOP_MODE','full')));
cfg=benchmark_config(mode);
oodRoot=getenv_default('AFOP_OOD18_ROOT',fullfile(root,'data','ood18'));
rawPath=fullfile(oodRoot,'dataset3.mat');labelsPath=fullfile(oodRoot,'labels_table.mat');
artifactPath=fullfile(cfg.paths.artifacts,sprintf('ood18_protocol_%s.mat',mode));
action=lower(string(getenv_default('AFOP_STAGE','prepare')));
comboIndex=str2double(getenv_default('AFOP_COMBO_INDEX','1'));
combos=[5 1 15;10 1 15;12 1 15;18 1 15; ...
        5 3 15;10 3 15;12 3 15;18 3 15];
episodes=episodes_by_mode(mode);

if action=="prepare"
    assert(isfile(rawPath) && isfile(labelsPath),'Missing OOD18 dataset or labels.');
    rawFile=load(rawPath,'dataset3');labelFile=load(labelsPath,'labels_table');
    assert(isequal(size(rawFile.dataset3),[18 60]),'OOD raw data must be 18 x 60.');
    features=build_target_feature_pool(rawFile.dataset3,labelFile.labels_table);
    manifest=generate_episode_manifest(features,(1:1080)',combos,episodes, ...
        cfg.seed.eval+28000000,"real_contact_ood18");
    if isfile(artifactPath)
        old=load(artifactPath,'manifest');assert(isequal(old.manifest,manifest), ...
            'Existing OOD manifest differs.');
    else
        ensure_dir(fileparts(artifactPath));
        save(artifactPath,'manifest','features','combos','rawPath','labelsPath','cfg','-v7.3');
    end
    fprintf('[OOD18] Prepared %d combinations x %d episodes.\n',size(combos,1),episodes);
    return;
end

assert(isfile(artifactPath),'Run AFOP_STAGE=prepare first.');
protocol=load(artifactPath,'manifest','features','rawPath');
assert(ismember(comboIndex,1:size(combos,1)),'Invalid combo index.');
manifest=subset_combo(protocol.manifest,comboIndex);
sourceProtocolPath=find_latest_protocol_artifact(cfg.paths.artifacts);
sourceProtocol=load(sourceProtocolPath,'frontend');
rows=(1:1080)';columns=sourceProtocol.frontend.selectedFeatures(:)';
runDir=fullfile(cfg.paths.runs,'ood18',char(action),sprintf('combo_%d',comboIndex));
ensure_dir(runDir);

switch action
    case "afop_eval"
        hp=struct('epochs',250,'learningRate',1.5e-3,'entropyWeight',0.10,'scale',1);
        result=evaluate_embeddings_on_manifest(single(protocol.features.X_all(:,columns)), ...
            rows,protocol.features,manifest,hp);
    case "afo_mlp_eval"
        result=evaluate_afo_mlp_manifest_checkpointed( ...
            single(protocol.features.X_all(:,columns)),rows,protocol.features,manifest,cfg, ...
            fullfile(runDir,'checkpoint.mat'),fullfile(runDir,'progress.log'),comboIndex);
    case {"feat_eval","mamlpp_eval","transformer_eval","gat_eval"}
        rawFile=load(protocol.rawPath,'dataset3');raw=rawFile.dataset3;
        if action=="feat_eval"
            shot=manifest.combos(1,2);modelPath=find_feat_model(cfg.paths.runs,shot,mode);
            model=load(modelPath,'encoderNet','adapterNet','normalization','hp','cfg');
            result=evaluate_feat_manifest(model.encoderNet,model.adapterNet,raw, ...
                protocol.features,manifest,model.normalization,model.hp,model.cfg);
        elseif action=="mamlpp_eval"
            modelPath=find_mamlpp_model(cfg.paths.runs,mode);
            model=load(modelPath,'params','normalization','cfg');
            params=move_mamlpp_parameters(model.params,cfg.device);
            result=evaluate_mamlpp_manifest(params,raw,protocol.features,manifest, ...
                model.normalization,model.cfg);
        else
            if action=="transformer_eval",family="tactile_transformer";else,family="channel_gat";end
            fold=struct('train_all',zeros(0,1),'test_all',rows);
            result=evaluate_support_only_raw_model_checkpointed(family,raw, ...
                protocol.features,fold,manifest,cfg,fullfile(runDir,'checkpoint.mat'), ...
                fullfile(runDir,'progress.log'),comboIndex);
        end
    case "cwt_prepare"
        rawFile=load(protocol.rawPath,'dataset3');
        imageRoot=fullfile(cfg.paths.artifacts,'ood18_cwt_images');
        existingCount=numel(dir(fullfile(imageRoot,'S*','*.jpg')));
        if existingCount==0
            create_cwt_images(rawFile.dataset3,imageRoot,cfg.paperCWT.inputSize, ...
                struct('waveletType',"amor",'voicesPerOctave',20));
        end
        assert(numel(dir(fullfile(imageRoot,'S*','*.jpg')))==1080, ...
            'OOD CWT image root is incomplete. Use a new empty output path.');
        modelPath=latest_cwt_model(cfg.paths.runs,mode);
        model=load(modelPath,'featureExtractor');
        embeddings=extract_cwt_embeddings(model.featureExtractor,imageRoot, ...
            protocol.features,rows,cfg.paperCWT,cfg.device);
        save(fullfile(cfg.paths.artifacts,'ood18_cwt_embeddings.mat'), ...
            'embeddings','rows','modelPath','imageRoot','-v7.3');
        fprintf('[OOD18:CWT] Prepared embeddings.\n');return;
    case "cwt_eval"
        cwt=load(fullfile(cfg.paths.artifacts,'ood18_cwt_embeddings.mat'), ...
            'embeddings','rows','modelPath');
        result=evaluate_cwt_manifest(cwt.embeddings,cwt.rows, ...
            protocol.features,manifest,cfg,fullfile(runDir,'checkpoint.mat'), ...
            fullfile(runDir,'progress.log'),comboIndex);
    otherwise
        error('Unknown AFOP_STAGE: %s',action);
end

outputPath=fullfile(runDir,'episodic.mat');
save_new_artifact(outputPath,struct('stage',action,'comboIndex',comboIndex, ...
    'manifest',manifest,'result',result,'sourceProtocolPath',sourceProtocolPath));
fprintf('[OOD18:%s] %dw%ds %.2f%% +/- %.2f\n',action,result.N_WAY, ...
    result.K_SHOT,100*result.acc_mean,100*result.acc_ci95);

function manifest=subset_combo(full,index)
manifest=full;manifest.combos=full.combos(index,:);manifest.entries=full.entries(index);
end
function path=find_feat_model(runsRoot,shot,mode)
files=dir(fullfile(runsRoot,"feat_official_raw_s"+shot+"_meta_*",'model.mat'));
path=latest_matching(files,mode);
end
function path=find_mamlpp_model(runsRoot,mode)
files=dir(fullfile(runsRoot,'mamlpp_raw_meta_*','model.mat'));path=latest_matching(files,mode);
end
function path=latest_matching(files,mode)
assert(~isempty(files),'Required source model is missing.');[~,order]=sort([files.datenum],'descend');path='';
fallback='';
for i=order
    candidate=fullfile(files(i).folder,files(i).name);s=load(candidate,'cfg');
    if string(s.cfg.mode)=="full",path=candidate;break;end
    if strlength(fallback)==0 && string(s.cfg.mode)==string(mode),fallback=candidate;end
end
if strlength(path)==0,path=fallback;end
assert(strlength(path)>0,'No full or mode-matched source model is available.');
end
function path=latest_cwt_model(runsRoot,mode)
files=dir(fullfile(runsRoot,'cwt_resnet50_source_train_*','model.mat'));
path=latest_matching(files,mode);
end
function n=episodes_by_mode(mode)
if mode=="smoke",n=3;elseif mode=="pilot",n=50;else,n=500;end
end
function ensure_dir(path)
if ~isfolder(path),mkdir(path);end
end
function value=getenv_default(name,fallback)
value=getenv(name);if isempty(value),value=fallback;end
end
