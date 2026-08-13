function encoderNet = build_feat_encoder(cfg)
% Build the four-block temporal FEAT encoder.

layers = imageInputLayer([1 cfg.data.signalLength cfg.data.numChannels], ...
    'Normalization','none', 'Name','raw_signal');
for block = 1:cfg.featOfficial.numBlocks
    layers = [layers
        convolution2dLayer([1 cfg.featOfficial.kernelSize], ...
            cfg.featOfficial.convChannels, 'Padding','same', ...
            'Name',sprintf('conv%d',block))
        batchNormalizationLayer('Name',sprintf('bn%d',block))
        reluLayer('Name',sprintf('relu%d',block))
        maxPooling2dLayer([1 2], 'Stride',[1 2], ...
            'Name',sprintf('pool%d',block))]; %#ok<AGROW>
end
layers = [layers
    globalMaxPooling2dLayer('Name','embedding')];
encoderNet = dlnetwork(layers);
end
