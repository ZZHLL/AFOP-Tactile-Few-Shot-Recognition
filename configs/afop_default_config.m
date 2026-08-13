function cfg = afop_default_config()
% Default configuration used by the closed-set AFOP pipeline.

cfg.fs = 1000;
cfg.numClasses = 36;
cfg.trialsPerClass = 60;
cfg.trainRatio = 0.5;
cfg.innerTrainRatio = 0.8;

cfg.seed.split = 20250812;
cfg.seed.dscan = 20250822;
cfg.seed.evalBase = 20250901;

cfg.jitter.windowSamples = 4000;
cfg.jitter.sigmaSamples = 200;
cfg.jitter.clipSamples = 600;
cfg.jitter.seed = 1;

cfg.dscan.candidates = 1:20;
cfg.dscan.way = 5;
cfg.dscan.shot = 5;
cfg.dscan.query = 1;
cfg.dscan.episodes = 300;

cfg.afop.numEpochs = 250;
cfg.afop.learningRate = 1.5e-3;
cfg.afop.entropyWeight = 0.10;
cfg.afop.scale = 1;

cfg.eval.combos = [5 1 15; 10 1 15; 12 1 15; 20 1 15; ...
    26 1 15; 28 1 15; 36 1 15; 5 3 15; 10 3 15; ...
    12 3 15; 20 3 15; 26 3 15; 28 3 15; 36 3 15; ...
    5 5 15; 10 5 15; 12 5 15; 36 5 15];
cfg.eval.episodes = 500;
end
