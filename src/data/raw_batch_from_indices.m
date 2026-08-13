function X = raw_batch_from_indices(raw, features, globalIndices, precision)
% Convert feature-row ids to a fixed C x T x B raw tensor.

if nargin < 4 || isempty(precision)
    precision = 'single';
end
globalIndices = globalIndices(:);
first = raw{features.y_class(globalIndices(1)), features.y_trial(globalIndices(1))};
X = zeros(size(first,1), size(first,2), numel(globalIndices), precision);
for i = 1:numel(globalIndices)
    row = globalIndices(i);
    c = features.y_class(row);
    t = features.y_trial(row);
    X(:,:,i) = cast(raw{c,t}, precision);
end
end

