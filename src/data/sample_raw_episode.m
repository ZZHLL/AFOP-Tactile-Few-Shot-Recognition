function episode = sample_raw_episode(raw, features, allowedRows, normalization, nWay, kShot, qShot, stream)
% Sample one raw-signal meta-training episode from an allowed row partition.

allowedRows = allowedRows(:);
y = features.y_class(:);
classes = unique(y(allowedRows))';
counts = arrayfun(@(c) nnz(y(allowedRows) == c), classes);
eligible = classes(counts >= kShot + qShot);
assert(numel(eligible) >= nWay, 'Insufficient eligible classes.');
classIds = eligible(randperm(stream, numel(eligible), nWay));
supportRows = zeros(nWay, kShot);
queryRows = zeros(nWay, qShot);
for ci = 1:nWay
    pool = allowedRows(y(allowedRows) == classIds(ci));
    order = randperm(stream, numel(pool), kShot + qShot);
    supportRows(ci,:) = pool(order(1:kShot));
    queryRows(ci,:) = pool(order(kShot+1:end));
end
episode = raw_episode_from_rows(raw, features, classIds, supportRows, queryRows, normalization);
end
