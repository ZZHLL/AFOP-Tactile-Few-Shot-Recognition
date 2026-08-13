function [lossValue, gradients] = mamlpp_meta_gradients(params, episode, cfg, enableHigherDerivatives)
% Compute query-based meta-gradients for the raw-signal MAML++ baseline.

[loss, gradients] = dlfeval(@meta_objective, params, episode, cfg, enableHigherDerivatives);
lossValue = double(gather(extractdata(loss)));
end

function [metaLoss, gradients] = meta_objective(params, episode, cfg, enableHigher)
support = raw_dlarray(episode.supportX, cfg.device);
query = raw_dlarray(episode.queryX, cfg.device);
fast = encoder_only(params);
nWay = numel(episode.classIds);
[headW, headB] = mamlpp_initialize_task_head(fast, support, ...
    episode.localSupport, nWay, cfg.mamlpp.strides);
weights = single(1:cfg.mamlpp.innerSteps);
weights = weights / sum(weights);
metaLoss = 0;
scale = struct('value',cfg.mamlpp.cosineScale,'strides',cfg.mamlpp.strides);
encoderNames = fieldnames(fast);

for step = 1:cfg.mamlpp.innerSteps
    [supportLoss, ~] = mamlpp_classification_loss(fast, headW, headB, ...
        support, episode.localSupport, scale);
    values = [struct2cell(fast); {headW}; {headB}];
    grads = cell(size(values));
    [grads{:}] = dlgradient(supportLoss, values{:}, ...
        'EnableHigherDerivatives',enableHigher);
    alpha = log(1 + exp(params.logInnerLR(step)));
    for i = 1:numel(encoderNames)
        fast.(encoderNames{i}) = fast.(encoderNames{i}) - alpha*grads{i};
    end
    headW = headW - alpha*grads{end-1};
    headB = headB - alpha*grads{end};
    [queryLoss, ~] = mamlpp_classification_loss(fast, headW, headB, ...
        query, episode.localQuery, scale);
    metaLoss = metaLoss + weights(step)*queryLoss;
end

paramNames = fieldnames(params);
paramValues = struct2cell(params);
gradientValues = cell(size(paramValues));
[gradientValues{:}] = dlgradient(metaLoss, paramValues{:});
gradients = cell2struct(gradientValues, paramNames, 1);
end

function fast = encoder_only(params)
fast = rmfield(params, 'logInnerLR');
end

function value = raw_dlarray(value, device)
value = reshape(single(value), size(value,1), size(value,2), 1, size(value,3));
if device == "gpu", value = gpuArray(value); end
value = dlarray(value, 'SSCB');
end
