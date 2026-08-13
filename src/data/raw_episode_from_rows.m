function episode = raw_episode_from_rows(raw, features, classIds, supportRows, queryRows, normalization)
% Build a normalized raw-signal episode from global row indices.

supportRows = reshape(supportRows', [], 1);
queryRows = reshape(queryRows', [], 1);
assert(isempty(intersect(supportRows, queryRows)), 'Support/query overlap.');

supportX = normalize_raw_tensor(raw_batch_from_indices(raw, features, supportRows), normalization);
queryX = normalize_raw_tensor(raw_batch_from_indices(raw, features, queryRows), normalization);
supportY = features.y_class(supportRows);
queryY = features.y_class(queryRows);
localSupport = zeros(numel(supportY),1);
localQuery = zeros(numel(queryY),1);
for ci = 1:numel(classIds)
    localSupport(supportY == classIds(ci)) = ci;
    localQuery(queryY == classIds(ci)) = ci;
end
assert(all(localSupport > 0) && all(localQuery > 0), 'Episode label mapping failed.');

episode.supportX = supportX;
episode.queryX = queryX;
episode.localSupport = localSupport;
episode.localQuery = localQuery;
episode.queryGlobalLabels = queryY;
episode.classIds = classIds(:)';
episode.supportRows = supportRows;
episode.queryRows = queryRows;
end
