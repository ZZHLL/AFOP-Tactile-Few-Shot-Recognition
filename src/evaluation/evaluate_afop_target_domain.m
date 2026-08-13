function result = evaluate_afop_target_domain(sourceFeatures, targetFeatures, ...
    sourceFitRows, sourceValidationRows, manifest, adaptationSeed, selection)
% Fit AFOP on source rows and evaluate fixed target-domain episodes.

fold = struct('train_all', [sourceFitRows(:); sourceValidationRows(:)], ...
    'source_train', sourceFitRows(:), ...
    'source_val', sourceValidationRows(:), 'test_all', manifest.allowedIdx(:));
assert(~isempty(fold.source_val), 'D-scan requires a nonempty source-validation split.');
assert(isempty(intersect(fold.source_train, fold.source_val)), ...
    'Source-training and source-validation rows overlap.');
cfg = afop_default_config();
if nargin >= 7 && ~isempty(selection)
    names = fieldnames(selection);
    for index = 1:numel(names)
        if strcmp(names{index}, 'seed')
            cfg.seed.dscan = selection.seed;
        else
            cfg.dscan.(names{index}) = selection.(names{index});
        end
    end
end

frontend = fit_nca_and_dscan(sourceFeatures, fold, cfg);
result = evaluate_afop(targetFeatures, frontend.selectedFeatures, ...
    manifest, cfg.afop, adaptationSeed);
for index = 1:numel(result)
    result(index).selectedD = frontend.selectedD;
end
end
