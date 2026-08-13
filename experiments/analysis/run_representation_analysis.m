clear; clc;
% Run CKA and t-SNE analysis on sample-aligned embeddings.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')),'-begin');
inputPath = string(getenv('AFOP_REPRESENTATION_INPUT'));
assert(strlength(inputPath)>0 && isfile(inputPath), ...
    'Set AFOP_REPRESENTATION_INPUT to a MAT file containing representations.');
loaded = load(inputPath);
assert(isfield(loaded,'representations') && isstruct(loaded.representations), ...
    'Input must contain a representations structure.');
names = string(fieldnames(loaded.representations));
sampleCount = size(loaded.representations.(names(1)),1);
assert(all(arrayfun(@(name)size(loaded.representations.(name),1)==sampleCount,names)), ...
    'All representations must use the same samples and order.');

pairwiseCKA = compute_pairwise_cka(loaded.representations);
outputRoot = fullfile(root,'artifacts','representation_analysis');
if ~isfolder(outputRoot),mkdir(outputRoot);end
stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
writetable(pairwiseCKA,fullfile(outputRoot,['pairwise_cka_' stamp '.csv']), ...
    'WriteRowNames',true);

tsneResults = struct();
for index = 1:numel(names)
    rng(20260311,'twister');
    tsneResults.(names(index)) = tsne(double(loaded.representations.(names(index))), ...
        'NumDimensions',2,'Standardize',true);
end

stability = table();
if isfield(loaded,'referenceRepresentations')
    conditionNames = string(fieldnames(loaded.referenceRepresentations));
    rows = strings(0,1); references = strings(0,1); values = zeros(0,1);
    for conditionIndex = 1:numel(conditionNames)
        condition = conditionNames(conditionIndex);
        current = loaded.referenceRepresentations.(condition);
        for modelIndex = 1:numel(names)
            model = names(modelIndex);
            assert(isfield(current,model),'Missing %s under %s.',model,condition);
            rows(end+1,1)=model; references(end+1,1)=condition; %#ok<SAGROW>
            values(end+1,1)=linear_cka(loaded.representations.(model),current.(model)); %#ok<SAGROW>
        end
    end
    stability = table(rows,references,values, ...
        'VariableNames',{'Model','Condition','CKAToReference'});
    writetable(stability,fullfile(outputRoot,['cka_to_reference_' stamp '.csv']));
end
save(fullfile(outputRoot,['representation_analysis_' stamp '.mat']), ...
    'pairwiseCKA','tsneResults','stability','names','-v7.3');
fprintf('[Representation analysis] Saved under %s\n',outputRoot);
