function result = evaluate_afo_mlp_manifest_checkpointed(embeddings,globalRows, ...
    features,manifest,cfg,checkpointPath,progressPath,reviewComboPosition)
% Evaluate one AFO-MLP combination with episode-level checkpointing.

assert(size(manifest.combos,1)==1,'Run one AFO-MLP combination at a time.');
combo=manifest.combos(1,:);
entries=manifest.entries{1};
globalRows=globalRows(:);
rowToLocal=zeros(max(globalRows),1);
rowToLocal(globalRows)=1:numel(globalRows);
state=initialize_state(combo,numel(entries),reviewComboPosition);
if isfile(checkpointPath)
    loaded=load(checkpointPath,'state');
    assert(isequal(loaded.state.combo,combo),'Checkpoint combo mismatch.');
    assert(loaded.state.reviewComboPosition==reviewComboPosition,'Checkpoint seed index mismatch.');
    state=loaded.state;
    append_progress(progressPath,sprintf('[resume] afo_mlp %dw%ds at %d/%d', ...
        combo(1),combo(2),state.completed,numel(entries)));
end

hp=cfg.paperHeads.mlp;
for ep=state.completed+1:numel(entries)
    entry=entries(ep);
    supportRows=reshape(entry.supportIdx',[],1);
    queryRows=reshape(entry.queryIdx',[],1);
    classIds=entry.classIds(:);
    supportX=double(embeddings(rowToLocal(supportRows),:));
    queryX=double(embeddings(rowToLocal(queryRows),:));
    supportY=features.y_class(supportRows);supportY=supportY(:);
    queryY=features.y_class(queryRows);queryY=queryY(:);
    [supportX,mu,sigma]=zscore(supportX);sigma(sigma==0)=1;
    queryX=(queryX-mu)./sigma;
    rng(cfg.seed.eval+2*1000000+reviewComboPosition*1000+ep,'twister');
    output=fit_predict_mlp(supportX,supportY,queryX,classIds,hp);
    state.accuracy(ep)=mean(output.predictions==queryY);
    state.trainMs(ep)=output.adaptationMs;
    state.inferMs(ep)=output.inferenceMs;
    state.completed=ep;
    message=sprintf('[afo_mlp] %dw%ds episode %d/%d | running %.2f%% | train %.1f s', ...
        combo(1),combo(2),ep,numel(entries),100*mean(state.accuracy(1:ep)), ...
        output.adaptationMs/1000);
    fprintf('%s\n',message);append_progress(progressPath,message);
    if mod(ep,cfg.paperRaw.checkpointEvery)==0 || ep==numel(entries)
        save_checkpoint(checkpointPath,state);
    end
end
result=struct('model','afo_mlp','N_WAY',combo(1),'K_SHOT',combo(2), ...
    'Q_SHOT',combo(3),'episodes',numel(entries),'acc_mean',mean(state.accuracy), ...
    'acc_std',std(state.accuracy),'acc_ci95',1.96*std(state.accuracy)/sqrt(numel(entries)), ...
    'adapt_ms_median',median(state.trainMs), ...
    'infer_ms_per_sample_median',median(state.inferMs/(combo(1)*combo(3))), ...
    'episode_accuracy',state.accuracy,'reviewComboPosition',reviewComboPosition);
end

function state=initialize_state(combo,numEpisodes,reviewComboPosition)
state=struct('combo',combo,'reviewComboPosition',reviewComboPosition,'completed',0, ...
    'accuracy',nan(numEpisodes,1),'trainMs',nan(numEpisodes,1),'inferMs',nan(numEpisodes,1));
end

function output=fit_predict_mlp(supportX,supportY,queryX,classIds,hp)
start=tic;
layers=[featureInputLayer(size(supportX,2),'Normalization','none')
    fullyConnectedLayer(hp.hiddenDim)
    reluLayer
    fullyConnectedLayer(hp.outputDim)
    reluLayer];
net=dlnetwork(layers);
support=dlarray(single(supportX'),'CB');
initial=gather(extractdata(predict(net,support)))';
weights0=zeros(numel(classIds),hp.outputDim,'single');
for i=1:numel(classIds)
    value=mean(initial(supportY==classIds(i),:),1);
    weights0(i,:)=value/(norm(value)+1e-9);
end
weights=dlarray(weights0);bias=dlarray(zeros(numel(classIds),1,'single'));
targets=one_hot(supportY,classIds);
avgNet=[];avgSqNet=[];avgW=[];avgSqW=[];avgB=[];avgSqB=[];
for epoch=1:hp.epochs
    [gradientNet,gradientW,gradientB]=dlfeval(@mlp_gradients,net,support,targets, ...
        weights,bias,hp.entropyWeight,hp.scale);
    [net.Learnables,avgNet,avgSqNet]=adamupdate(net.Learnables,gradientNet, ...
        avgNet,avgSqNet,epoch,hp.mlpLR);
    [weights,avgW,avgSqW]=adamupdate(weights,gradientW,avgW,avgSqW,epoch,hp.headLR);
    [bias,avgB,avgSqB]=adamupdate(bias,gradientB,avgB,avgSqB,epoch,hp.headLR);
end
output.adaptationMs=toc(start)*1000;
start=tic;
embedding=predict(net,dlarray(single(queryX'),'CB'));
probability=cosine_probability(embedding,weights,bias,hp.scale);
[~,index]=max(gather(extractdata(probability)),[],1);
output.predictions=reshape(classIds(index(:)),[],1);
output.inferenceMs=toc(start)*1000;
end

function [gradientNet,gradientW,gradientB]=mlp_gradients(net,input,targets,weights,bias,entropyWeight,scale)
embedding=predict(net,input);
probability=cosine_probability(embedding,weights,bias,scale);
crossEntropy=crossentropy(probability,targets,'DataFormat','CB');
entropy=-mean(sum(probability.*log(probability+1e-9),1));
loss=crossEntropy+entropyWeight*entropy;
[gradientNet,gradientW,gradientB]=dlgradient(loss,net.Learnables,weights,bias);
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
