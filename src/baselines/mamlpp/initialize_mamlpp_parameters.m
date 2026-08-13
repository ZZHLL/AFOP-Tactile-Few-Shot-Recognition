function params = initialize_mamlpp_parameters(inputChannels, cfg, device)
% Initialize a raw-signal CNN and learnable per-step inner rates.

channels = cfg.mamlpp.convChannels;
kernels = cfg.mamlpp.kernelSizes;
params.conv1W = dlarray(conv_glorot(kernels(1), inputChannels, channels(1)));
params.conv1B = dlarray(zeros(1, 1, channels(1), 'single'));
params.conv2W = dlarray(conv_glorot(kernels(2), channels(1), channels(2)));
params.conv2B = dlarray(zeros(1, 1, channels(2), 'single'));
params.conv3W = dlarray(conv_glorot(kernels(3), channels(2), channels(3)));
params.conv3B = dlarray(zeros(1, 1, channels(3), 'single'));
params.fcW = dlarray(glorot(cfg.mamlpp.embeddingDim, channels(3)));
params.fcB = dlarray(zeros(cfg.mamlpp.embeddingDim, 1, 'single'));
inverseSoftplus = log(exp(single(cfg.mamlpp.innerLR)) - 1);
params.logInnerLR = dlarray(repmat(inverseSoftplus, cfg.mamlpp.innerSteps, 1));
params = move_mamlpp_parameters(params, device);
end

function value = conv_glorot(kernelSize, inputChannels, outputChannels)
fanIn = kernelSize * inputChannels;
fanOut = kernelSize * outputChannels;
limit = sqrt(6 / (fanIn + fanOut));
value = single((2*rand(1, kernelSize, inputChannels, outputChannels) - 1) * limit);
end

function value = glorot(outputDim, inputDim)
limit = sqrt(6 / (inputDim + outputDim));
value = single((2*rand(outputDim, inputDim) - 1) * limit);
end
