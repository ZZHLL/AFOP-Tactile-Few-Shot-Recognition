clear; clc;
% Paired bootstrap CI, sign-flip permutation, and within-task Holm correction.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
inputPath = string(getenv('AFOP_STATISTICS_INPUT'));
assert(strlength(inputPath)>0 && isfile(inputPath), ...
    'Set AFOP_STATISTICS_INPUT to a MAT file containing comparisons.');
loaded = load(inputPath,'comparisons');
comparisons = loaded.comparisons;
assert(isstruct(comparisons) && all(isfield(comparisons, ...
    {'task','baseline','referenceAccuracy','baselineAccuracy'})), ...
    'comparisons must contain task, baseline, referenceAccuracy, baselineAccuracy.');
iterations = str2double(getenv_default('AFOP_STAT_ITERATIONS','20000'));
seed = str2double(getenv_default('AFOP_STAT_SEED','20260820'));

rows = repmat(struct('Task',"",'Baseline',"",'Episodes',0, ...
    'MeanDifferencePP',0,'CI95LowerPP',0,'CI95UpperPP',0, ...
    'PermutationP',0,'HolmAdjustedP',NaN),numel(comparisons),1);
for index = 1:numel(comparisons)
    reference = comparisons(index).referenceAccuracy(:);
    baseline = comparisons(index).baselineAccuracy(:);
    assert(numel(reference)==numel(baseline),'Paired episode count mismatch.');
    difference = reference-baseline;
    rng(seed+index,'twister');
    bootstrapMeans = zeros(iterations,1);
    for iteration = 1:iterations
        bootstrapMeans(iteration)=mean(difference(randi(numel(difference),numel(difference),1)));
    end
    bounds=prctile(bootstrapMeans,[2.5 97.5]);
    observed=abs(mean(difference)); extreme=0;
    for iteration = 1:iterations
        signs=2*(rand(numel(difference),1)>=0.5)-1;
        extreme=extreme+(abs(mean(signs.*difference))>=observed);
    end
    rows(index).Task=string(comparisons(index).task);
    rows(index).Baseline=string(comparisons(index).baseline);
    rows(index).Episodes=numel(difference);
    rows(index).MeanDifferencePP=100*mean(difference);
    rows(index).CI95LowerPP=100*bounds(1);
    rows(index).CI95UpperPP=100*bounds(2);
    rows(index).PermutationP=(extreme+1)/(iterations+1);
end

tasks=unique(string({rows.Task}));
for taskIndex=1:numel(tasks)
    indices=find(string({rows.Task})==tasks(taskIndex));
    adjusted=holm_adjust([rows(indices).PermutationP]');
    for localIndex=1:numel(indices)
        rows(indices(localIndex)).HolmAdjustedP=adjusted(localIndex);
    end
end
resultTable=struct2table(rows);
outputRoot=fullfile(root,'artifacts','statistics');
if ~isfolder(outputRoot),mkdir(outputRoot);end
stamp=char(datetime('now','Format','yyyyMMdd_HHmmss'));
writetable(resultTable,fullfile(outputRoot,['paired_statistics_' stamp '.csv']));

function adjusted=holm_adjust(pValues)
[sorted,order]=sort(pValues,'ascend'); count=numel(sorted);
adjustedSorted=zeros(count,1); running=0;
for index=1:count
    running=max(running,(count-index+1)*sorted(index));
    adjustedSorted(index)=min(1,running);
end
adjusted=zeros(count,1); adjusted(order)=adjustedSorted;
end
