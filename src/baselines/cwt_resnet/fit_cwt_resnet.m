function [featureExtractor,info] = fit_cwt_resnet(imageRoot,features,fold,hp,device,seed)
% Train the CWT-ResNet50 baseline using feature-table row mapping.

rng(seed,'twister');
trainRows=fold.train_all(:);
[files,labels]=cwt_files_from_rows(imageRoot,features,trainRows);
assert(all(isfile(files)),'One or more CWT images are missing.');
imds=imageDatastore(cellstr(files),'Labels',categorical(labels));
[imdsTrain,imdsVal]=splitEachLabel(imds,0.9,'randomized');

network=resnet50;
graph=layerGraph(network);
graph=removeLayers(graph,{network.Layers(end-2).Name,network.Layers(end-1).Name, ...
    network.Layers(end).Name});
numClasses=numel(unique(labels));
head=[fullyConnectedLayer(numClasses,'Name','fc_new', ...
        'WeightLearnRateFactor',hp.headLR/hp.backboneLR, ...
        'BiasLearnRateFactor',hp.headLR/hp.backboneLR)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
graph=addLayers(graph,head);
graph=connectLayers(graph,'avg_pool','fc_new');

names=string({graph.Layers.Name});
freezeIndex=find(names==string(hp.freezeUntil),1);
assert(~isempty(freezeIndex),'Unable to find freeze layer %s.',hp.freezeUntil);
for i=1:freezeIndex
    layer=graph.Layers(i);
    if isprop(layer,'WeightLearnRateFactor'),layer.WeightLearnRateFactor=0;end
    if isprop(layer,'BiasLearnRateFactor'),layer.BiasLearnRateFactor=0;end
    graph=replaceLayer(graph,layer.Name,layer);
end

validationFrequency=max(1,floor(numel(imdsTrain.Files)/hp.miniBatch));
options=trainingOptions('adam','InitialLearnRate',hp.backboneLR, ...
    'MiniBatchSize',hp.miniBatch,'MaxEpochs',hp.maxEpochs, ...
    'Shuffle','every-epoch','Verbose',true,'Plots','none', ...
    'ExecutionEnvironment',char(device),'ValidationData',imdsVal, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',hp.earlyStopPatience);
trained=trainNetwork(imdsTrain,graph,options);
featureGraph=layerGraph(trained);
featureGraph=removeLayers(featureGraph,{'fc_new','softmax','classification'});
featureExtractor=dlnetwork(featureGraph);
if string(device)=="gpu"
    featureExtractor=dlupdate(@gpuArray,featureExtractor);
end
info=struct('trainRows',trainRows,'trainFiles',{imdsTrain.Files}, ...
    'valFiles',{imdsVal.Files},'seed',seed,'imageRoot',char(imageRoot), ...
    'numTrain',numel(imdsTrain.Files),'numVal',numel(imdsVal.Files));
end
