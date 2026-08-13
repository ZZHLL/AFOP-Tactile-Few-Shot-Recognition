function results = evaluate_prototype_initialization(embeddings,globalRows, ...
    features,manifest)
% Evaluate support prototypes without episode-time gradient updates.

globalRows = globalRows(:);
rowToLocal = zeros(max(globalRows),1);
rowToLocal(globalRows) = 1:numel(globalRows);
template = struct('NWay',[],'KShot',[],'QShot',[],'episodes',[], ...
    'accMean',[],'accStd',[],'ci95',[],'episodeAccuracy',[]);
results = repmat(template,size(manifest.combos,1),1);
for comboIndex = 1:size(manifest.combos,1)
    entries = manifest.entries{comboIndex};
    accuracy = zeros(numel(entries),1);
    for episode = 1:numel(entries)
        entry = entries(episode);
        supportRows = reshape(entry.supportIdx',[],1);
        queryRows = reshape(entry.queryIdx',[],1);
        supportX = embeddings(rowToLocal(supportRows),:);
        queryX = embeddings(rowToLocal(queryRows),:);
        supportY = features.y_class(supportRows);
        queryY = features.y_class(queryRows);
        [supportX,mu,sigma] = zscore(supportX); sigma(sigma==0)=1;
        queryX = (queryX-mu)./sigma;
        supportX = normalize(supportX,2); queryX=normalize(queryX,2);
        prototypes = zeros(numel(entry.classIds),size(supportX,2));
        for classIndex = 1:numel(entry.classIds)
            p=mean(supportX(supportY==entry.classIds(classIndex),:),1);
            prototypes(classIndex,:)=p/(norm(p)+1e-9);
        end
        [~,index]=max(queryX*prototypes',[],2);
        prediction=entry.classIds(index(:));
        accuracy(episode)=mean(prediction(:)==queryY(:));
    end
    combo=manifest.combos(comboIndex,:);
    item=template; item.NWay=combo(1); item.KShot=combo(2); item.QShot=combo(3);
    item.episodes=numel(accuracy); item.accMean=mean(accuracy);
    item.accStd=std(accuracy); item.ci95=1.96*std(accuracy)/sqrt(numel(accuracy));
    item.episodeAccuracy=accuracy; results(comboIndex)=item;
end
end
