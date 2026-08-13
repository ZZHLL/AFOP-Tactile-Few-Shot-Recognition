function [headW, headB] = mamlpp_initialize_task_head(encoderParams, support, localSupport, nWay, strides)
% Initialize a differentiable task-local head from support prototypes.

embedding = mamlpp_encoder(encoderParams, support, strides);
parts = cell(1,nWay);
for ci = 1:nWay
    parts{ci} = mean(embedding(:, localSupport == ci), 2)';
end
headW = cat(1, parts{:});
base = extractdata(encoderParams.fcB);
headB = zeros(nWay, 1, 'like',base) + 0*sum(encoderParams.fcB(:));
end
