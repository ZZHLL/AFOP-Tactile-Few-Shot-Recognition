function manifest = generate_episode_manifest(features, allowedIdx, combos, episodes, seed, purpose)
% Generate shared support/query row indices for every model.

if nargin < 6 || isempty(purpose)
    purpose = "evaluation";
end
stream = RandStream('mt19937ar', 'Seed', seed);
y = features.y_class(:);
allowedIdx = allowedIdx(:);

manifest = struct();
manifest.version = 1;
manifest.purpose = string(purpose);
manifest.seed = seed;
manifest.allowedIdx = allowedIdx;
manifest.combos = combos;
manifest.entries = cell(size(combos,1),1);

availableClasses = unique(y(allowedIdx))';
for comboIdx = 1:size(combos,1)
    nWay = combos(comboIdx,1);
    kShot = combos(comboIdx,2);
    qShot = combos(comboIdx,3);
    counts = arrayfun(@(c) nnz(y(allowedIdx) == c), availableClasses);
    eligible = availableClasses(counts >= kShot + qShot);
    assert(numel(eligible) >= nWay, 'Not enough eligible classes for %d-way.', nWay);

    entries = repmat(struct('classIds',[],'supportIdx',[],'queryIdx',[]), episodes, 1);
    for ep = 1:episodes
        order = randperm(stream, numel(eligible), nWay);
        classIds = eligible(order);
        supportIdx = zeros(nWay, kShot);
        queryIdx = zeros(nWay, qShot);
        for ci = 1:nWay
            pool = allowedIdx(y(allowedIdx) == classIds(ci));
            pick = randperm(stream, numel(pool), kShot + qShot);
            supportIdx(ci,:) = pool(pick(1:kShot));
            queryIdx(ci,:) = pool(pick(kShot+1:end));
        end
        entries(ep).classIds = classIds;
        entries(ep).supportIdx = supportIdx;
        entries(ep).queryIdx = queryIdx;
    end
    manifest.entries{comboIdx} = entries;
end

validate_episode_manifest(manifest, features);
end
