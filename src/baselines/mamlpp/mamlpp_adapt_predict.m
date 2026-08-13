function [probability, adaptationMs, inferenceMs] = mamlpp_adapt_predict(params, episode, cfg)
% Adapt the raw-signal CNN on support and classify the query set.

start = tic;
[adapted, headW, headB] = dlfeval(@adapt_task, params, episode, cfg);
adaptationMs = toc(start)*1000;
query = raw_dlarray(episode.queryX, cfg.device);
scale = struct('value',cfg.mamlpp.cosineScale,'strides',cfg.mamlpp.strides);
start = tic;
[~, probability] = mamlpp_classification_loss(adapted, headW, headB, ...
    query, episode.localQuery, scale);
inferenceMs = toc(start)*1000;
end

function [fast, headW, headB] = adapt_task(params, episode, cfg)
support = raw_dlarray(episode.supportX, cfg.device);
fast = rmfield(params, 'logInnerLR');
nWay = numel(episode.classIds);
[headW, headB] = mamlpp_initialize_task_head(fast, support, ...
    episode.localSupport, nWay, cfg.mamlpp.strides);
scale = struct('value',cfg.mamlpp.cosineScale,'strides',cfg.mamlpp.strides);
encoderNames = fieldnames(fast);
for step = 1:cfg.mamlpp.innerSteps
    [supportLoss, ~] = mamlpp_classification_loss(fast, headW, headB, ...
        support, episode.localSupport, scale);
    values = [struct2cell(fast); {headW}; {headB}];
    grads = cell(size(values));
    [grads{:}] = dlgradient(supportLoss, values{:});
    alpha = log(1 + exp(params.logInnerLR(step)));
    for i = 1:numel(encoderNames)
        fast.(encoderNames{i}) = fast.(encoderNames{i}) - alpha*grads{i};
    end
    headW = headW - alpha*grads{end-1};
    headB = headB - alpha*grads{end};
end
end

function value = raw_dlarray(value, device)
value = reshape(single(value), size(value,1), size(value,2), 1, size(value,3));
if device == "gpu", value = gpuArray(value); end
value = dlarray(value, 'SSCB');
end
