function episode = raw_episode_from_manifest(raw, features, entry, normalization)
% Materialize one stored evaluation episode as raw four-channel signals.

episode = raw_episode_from_rows(raw, features, entry.classIds, ...
    entry.supportIdx, entry.queryIdx, normalization);
end
