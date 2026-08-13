function cfg = benchmark_config(mode)
% Configuration shared by the four raw-signal baselines and evaluations.

if nargin < 1 || isempty(mode), mode = "smoke"; end
mode = lower(string(mode));
assert(any(mode == ["smoke","pilot","full"]));

cfg.mode = mode;
cfg.root = fileparts(fileparts(mfilename('fullpath')));
dataRoot = fullfile(cfg.root,'data');
cfg.input.dataset = fullfile(dataRoot,'dataset2.mat');
cfg.input.labels = fullfile(dataRoot,'labels_table.mat');
cfg.input.features = fullfile(dataRoot,'features_with_labels.mat');
cfg.paths.artifacts = fullfile(cfg.root,'artifacts');
cfg.paths.runs = fullfile(cfg.root,'runs');
cfg.paths.logs = fullfile(cfg.root,'logs');

cfg.seed.fold = 20250812;
cfg.seed.dscan = 20250822;
cfg.seed.eval = 20260820;
cfg.seed.train = 20260830;
cfg.fold.trainRatio = 0.5;
cfg.fold.innerTrainRatio = 0.8;

cfg.dscan.candidates = 1:20;
cfg.dscan.way = 5;
cfg.dscan.shot = 5;
cfg.dscan.query = 1;
cfg.dscan.episodes = 300;

cfg.eval.query = 15;
cfg.eval.combos = [5 1 15;10 1 15;12 1 15;20 1 15;26 1 15;28 1 15;36 1 15; ...
    5 3 15;10 3 15;12 3 15;20 3 15;26 3 15;28 3 15;36 3 15; ...
    5 5 15;10 5 15;12 5 15;36 5 15];
cfg.eval.episodes = choose_by_mode(mode,3,50,500);
cfg.device = "gpu";
cfg.data = struct('numClasses',36,'trialsPerClass',60,'numChannels',4, ...
    'signalLength',4000,'samplingRate',1000);
cfg.preprocess.channelNames = ["PVDF1","PVDF2","SG1","SG2"];
cfg.preprocess.modalityIds = [1 1 2 2];

cfg.paperHeads.mlp = struct('epochs',1000,'mlpLR',1e-4, ...
    'headLR',1.5e-3,'entropyWeight',0.025,'outputDim',32, ...
    'hiddenDim',64,'scale',1);
cfg.paperRaw.checkpointEvery = 5;

cfg.paperCWT.imageRoot = fullfile(dataRoot,'CWTImages','200_4s');
cfg.paperCWT.backboneLR = 1e-5;
cfg.paperCWT.headLR = 1e-3;
cfg.paperCWT.maxEpochs = 8;
cfg.paperCWT.miniBatch = 32;
cfg.paperCWT.earlyStopPatience = 2;
cfg.paperCWT.freezeUntil = 'activation_22_relu';
cfg.paperCWT.inputSize = [224 224];
cfg.paperCWT.episodeEpochs = 150;
cfg.paperCWT.episodeLR = 1e-3;
cfg.paperCWT.entropyWeight = 0.10;
cfg.paperCWT.scale = 1;

cfg.transformer.embeddingDim = 64;
cfg.transformer.numHeads = 4;
cfg.transformer.dropout = 0.10;
cfg.transformer.supportOnlyEpochs = choose_by_mode(mode,1,12,25);
cfg.transformer.supportOnlyLR = 1e-3;

cfg.gat.temporalBins = 128;
cfg.gat.embeddingDim = 64;
cfg.gat.numHeads = 4;
cfg.gat.dropout = 0.10;
cfg.gat.supportOnlyEpochs = choose_by_mode(mode,1,15,30);
cfg.gat.supportOnlyLR = 1e-3;

cfg.featOfficial.embeddingDim = 64;
cfg.featOfficial.convChannels = 64;
cfg.featOfficial.numBlocks = 4;
cfg.featOfficial.kernelSize = 3;
cfg.featOfficial.attentionScoreDropout = 0.1;
cfg.featOfficial.attentionOutputDropout = 0.5;
cfg.featOfficial.metaWay = 5;
cfg.featOfficial.metaQuery = 15;
cfg.featOfficial.pretrainBatchSize = 16;
cfg.featOfficial.pretrainEpochs = choose_by_mode(mode,2,30,500);
cfg.featOfficial.pretrainLR = 1e-3;
cfg.featOfficial.pretrainWeightDecay = 5e-4;
cfg.featOfficial.pretrainSchedule = [75 150 300];
cfg.featOfficial.pretrainGamma = 0.1;
cfg.featOfficial.pretrainValidationEvery = choose_by_mode(mode,1,5,5);
cfg.featOfficial.pretrainDenseValidationAfter = inf;
cfg.featOfficial.pretrainValidationEpisodes = choose_by_mode(mode,2,25,200);
cfg.featOfficial.metaEpochs = choose_by_mode(mode,2,20,200);
cfg.featOfficial.episodesPerEpoch = choose_by_mode(mode,2,25,100);
cfg.featOfficial.metaLR = 1e-4;
cfg.featOfficial.adapterLRMultiplier = 10;
cfg.featOfficial.metaStepEpochs = 20;
cfg.featOfficial.metaGamma = 0.5;
cfg.featOfficial.metaValidationEvery = 1;
cfg.featOfficial.metaValidationEpisodes = choose_by_mode(mode,2,25,200);
cfg.featOfficial.shots = [1 3 5];
cfg.featOfficial.balanceByShot = [1.0 0.1 0.1];
cfg.featOfficial.temperatureByShot = [64 32 32];
cfg.featOfficial.temperature2ByShot = [16 64 64];
cfg.featOfficial.officialCommit = "47bdc7c1672e00b027c67469d0291e7502918950";

cfg.mamlpp.metaWay = 5;
cfg.mamlpp.metaShot = 5;
cfg.mamlpp.metaQuery = 10;
cfg.mamlpp.metaBatch = choose_by_mode(mode,1,2,4);
cfg.mamlpp.innerSteps = 5;
cfg.mamlpp.iterations = choose_by_mode(mode,1,200,2000);
cfg.mamlpp.outerLR = 1e-3;
cfg.mamlpp.innerLR = 1e-2;
cfg.mamlpp.hiddenDim = 128;
cfg.mamlpp.embeddingDim = 64;
cfg.mamlpp.cosineScale = 10;
cfg.mamlpp.firstOrderFraction = 0.40;
cfg.mamlpp.validationEvery = choose_by_mode(mode,1,20,100);
cfg.mamlpp.validationEpisodes = choose_by_mode(mode,2,25,100);
cfg.mamlpp.inputKind = "raw_signal";
cfg.mamlpp.convChannels = [16 32 64];
cfg.mamlpp.kernelSizes = [51 25 11];
cfg.mamlpp.strides = [4 4 2];
end

function value = choose_by_mode(mode,smokeValue,pilotValue,fullValue)
if mode == "smoke", value = smokeValue;
elseif mode == "pilot", value = pilotValue;
else, value = fullValue;
end
end
