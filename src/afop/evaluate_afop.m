function results = evaluate_afop(features,selectedFeatures,manifest,hp,seedBase)
% Evaluate prototype-initialized cosine-softmax adaptation on fixed episodes.

X = features.X_all(:,selectedFeatures);
y = features.y_class(:);
template = struct('NWay',[],'KShot',[],'QShot',[],'episodes',[], ...
    'accMean',[],'accStd',[],'ci95',[],'adaptMsMedian',[], ...
    'inferMsPerSampleMedian',[],'episodeAccuracy',[]);
results = repmat(template,size(manifest.combos,1),1);

for k = 1:size(manifest.combos,1)
    combo = manifest.combos(k,:); entries = manifest.entries{k};
    acc = zeros(numel(entries),1); adaptMs = acc; inferMs = acc;
    for ep = 1:numel(entries)
        e = entries(ep);
        supportRows = reshape(e.supportIdx',[],1);
        queryRows = reshape(e.queryIdx',[],1);
        supportX = X(supportRows,:); queryX = X(queryRows,:);
        [supportX,mu,sigma] = zscore(supportX); sigma(sigma==0)=1;
        queryX = (queryX-mu)./sigma;
        rng(seedBase+1000*k+ep,'twister');
        out = adapt_head(supportX,y(supportRows),queryX,e.classIds,hp);
        acc(ep) = mean(out.predictions==y(queryRows));
        adaptMs(ep) = out.adaptationMs; inferMs(ep) = out.inferenceMs;
    end
    r = template; r.NWay=combo(1); r.KShot=combo(2); r.QShot=combo(3);
    r.episodes=numel(entries); r.accMean=mean(acc); r.accStd=std(acc);
    r.ci95=1.96*r.accStd/sqrt(r.episodes); r.adaptMsMedian=median(adaptMs);
    r.inferMsPerSampleMedian=median(inferMs/(combo(1)*combo(3)));
    r.episodeAccuracy=acc; results(k)=r;
end
end

function out = adapt_head(supportX,supportY,queryX,classIds,hp)
t = tic; n = numel(classIds);
weights0 = zeros(n,size(supportX,2),'single');
for i = 1:n
    p = mean(supportX(supportY==classIds(i),:),1);
    weights0(i,:) = p/(norm(p)+1e-9);
end
weights = dlarray(weights0); bias = dlarray(zeros(n,1,'single'));
targets = zeros(n,numel(supportY),'single');
for i = 1:n, targets(i,supportY==classIds(i))=1; end
targets = dlarray(targets,'CB'); support = dlarray(single(supportX'),'CB');
avgW=[];avgSqW=[];avgB=[];avgSqB=[];
for epoch = 1:hp.numEpochs
    [gW,gB] = dlfeval(@gradients,support,targets,weights,bias,hp);
    [weights,avgW,avgSqW] = adamupdate(weights,gW,avgW,avgSqW,epoch,hp.learningRate);
    [bias,avgB,avgSqB] = adamupdate(bias,gB,avgB,avgSqB,epoch,hp.learningRate);
end
out.adaptationMs = toc(t)*1000;
t = tic; probability = cosine_probability(dlarray(single(queryX'),'CB'),weights,bias,hp.scale);
[~,idx] = max(gather(extractdata(probability)),[],1);
out.predictions = reshape(classIds(idx),[],1); out.inferenceMs = toc(t)*1000;
end

function [gW,gB] = gradients(input,targets,weights,bias,hp)
p = cosine_probability(input,weights,bias,hp.scale);
ce = crossentropy(p,targets,'DataFormat','CB');
entropy = -mean(sum(p.*log(p+1e-9),1));
[gW,gB] = dlgradient(ce+hp.entropyWeight*entropy,weights,bias);
end

function p = cosine_probability(input,weights,bias,scale)
w = weights./(vecnorm(weights,2,2)+1e-6);
x = input./(vecnorm(input,2,1)+1e-6);
p = softmax(scale*(stripdims(w)*stripdims(x))+bias,'DataFormat','CB');
end
