function subset = subset_episode_manifest(manifest, requestedCombos, numEpisodes, features)
% Select named combinations and an episode prefix without resampling.

requestedCombos = double(requestedCombos);
subset = manifest;
subset.combos = requestedCombos;
subset.entries = cell(size(requestedCombos,1),1);
for i = 1:size(requestedCombos,1)
    match = find(all(manifest.combos == requestedCombos(i,:), 2), 1);
    assert(~isempty(match), 'Requested combo [%s] is absent from the formal manifest.', ...
        num2str(requestedCombos(i,:)));
    available = manifest.entries{match};
    assert(numel(available) >= numEpisodes, 'Not enough stored episodes.');
    subset.entries{i} = available(1:numEpisodes);
end
subset.numEpisodes = numEpisodes;
subset.parentPurpose = manifest.purpose;
subset.purpose = manifest.purpose + "_prefix_" + string(numEpisodes);
validate_episode_manifest(subset, features);
end
