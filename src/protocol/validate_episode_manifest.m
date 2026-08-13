function validate_episode_manifest(manifest, features)
% Verify class labels, disjoint support/query sets, and allowed-index bounds.

y = features.y_class(:);
allowed = manifest.allowedIdx(:);
for comboIdx = 1:size(manifest.combos,1)
    combo = manifest.combos(comboIdx,:);
    entries = manifest.entries{comboIdx};
    for ep = 1:numel(entries)
        e = entries(ep);
        assert(isequal(size(e.supportIdx), [combo(1), combo(2)]), 'Bad support shape.');
        assert(isequal(size(e.queryIdx), [combo(1), combo(3)]), 'Bad query shape.');
        assert(isempty(intersect(e.supportIdx(:), e.queryIdx(:))), 'Support/query overlap.');
        assert(all(ismember([e.supportIdx(:); e.queryIdx(:)], allowed)), 'Episode uses forbidden rows.');
        for ci = 1:combo(1)
            assert(all(y(e.supportIdx(ci,:)) == e.classIds(ci)), 'Support class mismatch.');
            assert(all(y(e.queryIdx(ci,:)) == e.classIds(ci)), 'Query class mismatch.');
        end
    end
end
end

