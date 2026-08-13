function result=evaluate_cwt_manifest(embeddings,globalRows,features,manifest, ...
    cfg,checkpointPath,progressPath,reviewComboPosition)
% Evaluate one CWT-ResNet combination with the episodic head.

assert(size(manifest.combos,1)==1,'Run one CWT combination at a time.');
combo=manifest.combos(1,:);entries=manifest.entries{1};
globalRows=globalRows(:);rowToLocal=zeros(max(globalRows),1);
rowToLocal(globalRows)=1:numel(globalRows);
state=struct('combo',combo,'reviewComboPosition',reviewComboPosition,'completed',0, ...
    'accuracy',nan(numel(entries),1),'trainMs',nan(numel(entries),1), ...
    'inferMs',nan(numel(entries),1));
if isfile(checkpointPath)
    loaded=load(checkpointPath,'state');
    assert(isequal(loaded.state.combo,combo),'Checkpoint combo mismatch.');
    state=loaded.state;
end
hp=cfg.paperCWT;
for ep=state.completed+1:numel(entries)
    entry=entries(ep);classIds=entry.classIds(:);
    supportRows=reshape(entry.supportIdx',[],1);
    queryRows=reshape(entry.queryIdx',[],1);
    supportX=double(embeddings(rowToLocal(supportRows),:));
    queryX=double(embeddings(rowToLocal(queryRows),:));
    supportY=features.y_class(supportRows);supportY=supportY(:);
    queryY=features.y_class(queryRows);queryY=queryY(:);
    output=fit_predict_head(supportX,supportY,queryX,classIds,hp);
    state.accuracy(ep)=mean(output.predictions==queryY);
    state.trainMs(ep)=output.adaptationMs;state.inferMs(ep)=output.inferenceMs;
    state.completed=ep;
    message=sprintf('[cwt_resnet50] %dw%ds episode %d/%d | running %.2f%%', ...
        combo(1),combo(2),ep,numel(entries),100*mean(state.accuracy(1:ep)));
    fprintf('%s\n',message);append_progress(progressPath,message);
    if mod(ep,10)==0 || ep==numel(entries),save_checkpoint(checkpointPath,state);end
end
result=struct('model','cwt_resnet50','N_WAY',combo(1),'K_SHOT',combo(2), ...
    'Q_SHOT',combo(3),'episodes',numel(entries),'acc_mean',mean(state.accuracy), ...
    'acc_std',std(state.accuracy),'acc_ci95',1.96*std(state.accuracy)/sqrt(numel(entries)), ...
    'adapt_ms_median',median(state.trainMs), ...
    'infer_ms_per_sample_median',median(state.inferMs/(combo(1)*combo(3))), ...
    'episode_accuracy',state.accuracy,'reviewComboPosition',reviewComboPosition);
end

function output=fit_predict_head(supportX,supportY,queryX,classIds,hp)
start=tic;numClasses=numel(classIds);
weights0=zeros(numClasses,size(supportX,2),'single');
for i=1:numClasses
    value=mean(supportX(supportY==classIds(i),:),1);
    weights0(i,:)=value/(norm(value)+1e-9);
end
weights=dlarray(weights0);bias=dlarray(zeros(numClasses,1,'single'));
support=dlarray(single(supportX'),'CB');targets=one_hot(supportY,classIds);
avgW=[];avgSqW=[];avgB=[];avgSqB=[];
for epoch=1:hp.episodeEpochs
    [gradientW,gradientB]=dlfeval(@head_gradients,support,targets,weights,bias, ...
        hp.entropyWeight,hp.scale);
    [weights,avgW,avgSqW]=adamupdate(weights,gradientW,avgW,avgSqW,epoch,hp.episodeLR);
    [bias,avgB,avgSqB]=adamupdate(bias,gradientB,avgB,avgSqB,epoch,hp.episodeLR);
end
output.adaptationMs=toc(start)*1000;start=tic;
probability=cosine_probability(dlarray(single(queryX'),'CB'),weights,bias,hp.scale);
[~,index]=max(gather(extractdata(probability)),[],1);
output.predictions=reshape(classIds(index(:)),[],1);output.inferenceMs=toc(start)*1000;
end

function [gradientW,gradientB]=head_gradients(input,targets,weights,bias,entropyWeight,scale)
probability=cosine_probability(input,weights,bias,scale);
crossEntropy=crossentropy(probability,targets,'DataFormat','CB');
entropy=-mean(sum(probability.*log(probability+1e-9),1));
loss=crossEntropy+entropyWeight*entropy;
[gradientW,gradientB]=dlgradient(loss,weights,bias);
end

function probability=cosine_probability(input,weights,bias,scale)
normalWeights=weights./(vecnorm(weights,2,2)+1e-6);
normalInput=input./(vecnorm(input,2,1)+1e-6);
probability=softmax(scale*(stripdims(normalWeights)*stripdims(normalInput))+bias, ...
    'DataFormat','CB');
end

function targets=one_hot(labels,classIds)
targets=zeros(numel(classIds),numel(labels),'single');
for i=1:numel(classIds),targets(i,labels==classIds(i))=1;end
targets=dlarray(targets,'CB');
end

function save_checkpoint(path,state)
temporaryPath=[path '.tmp.mat'];save(temporaryPath,'state','-v7.3');movefile(temporaryPath,path,'f');
end

function append_progress(path,message)
fid=fopen(path,'a');assert(fid>=0,'Unable to open progress log: %s',path);
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%s | %s\n',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')),message);
end
