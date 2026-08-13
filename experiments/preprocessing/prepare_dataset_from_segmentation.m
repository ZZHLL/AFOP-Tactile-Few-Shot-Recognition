clear; clc;
% Continue from the timestamped segmentation artifact to AFOP-ready files.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src')),'-begin');
addpath(root,'-begin');

segmentationPath = string(getenv('AFOP_SEGMENTATION_ARTIFACT'));
assert(strlength(segmentationPath)>0 && isfile(segmentationPath), ...
    'Set AFOP_SEGMENTATION_ARTIFACT to a run_auto_segmentation output file.');
outputRoot = string(getenv_default('AFOP_PREPARED_OUTPUT',fullfile(root,'data')));
if ~isfolder(outputRoot), mkdir(outputRoot); end

referenceChannel = parse_integer('AFOP_CENTER_CHANNEL',3);
windowSamples = parse_integer('AFOP_WINDOW_SAMPLES',4000);
sigmaSamples = parse_integer('AFOP_SHIFT_SIGMA',200);
clipSamples = parse_integer('AFOP_SHIFT_CLIP',600);
jitterSeed = parse_integer('AFOP_JITTER_SEED',1);
roiText = string(getenv('AFOP_CENTER_ROI'));

source = load(segmentationPath,'shapeDataSet');
shapeDataSet = source.shapeDataSet;
assert(all(string({shapeDataSet.status})=="ok"), ...
    'At least one recording failed automatic segmentation.');
numClasses = numel(shapeDataSet);
assert(isfield(shapeDataSet,'fullData'), ...
    'The segmentation artifact does not contain baseline-corrected fullData.');
numTrials = size(shapeDataSet(1).fullData,3);
datasetFullTrials = cell(numClasses,numTrials);
centerIndices = zeros(numClasses,1);

for classId = 1:numClasses
    fullData = shapeDataSet(classId).fullData;
    assert(ndims(fullData)==3 && size(fullData,2)==4 && ...
        size(fullData,3)==numTrials, ...
        'Expected fullData to be time x 4 channels x %d trials for class %d.', ...
        numTrials,classId);
    for trialId = 1:numTrials
        trial = permute(fullData(:,:,trialId),[2 1 3]);
        datasetFullTrials{classId,trialId} = reshape(trial,4,size(fullData,1));
    end

    if isfield(shapeDataSet,'grooveCenterIdx') && ...
            ~isempty(shapeDataSet(classId).grooveCenterIdx)
        centerIndices(classId) = round(shapeDataSet(classId).grooveCenterIdx);
        continue;
    end
    archetype = shapeDataSet(classId).archetype;
    if isvector(archetype)
        archetype = squeeze(mean(shapeDataSet(classId).fullData,3,'omitnan'));
    end
    if isempty(roiText)
        roi = [1 size(archetype,1)];
    else
        roi = sscanf(roiText,'%d,%d')';
        assert(numel(roi)==2,'AFOP_CENTER_ROI must be first,last.');
    end
    centerIndices(classId) = estimate_nominal_center(archetype,referenceChannel,roi);
end

% Keep both endpoints so sample 4001 is the exact nominal center.
alignedSamples = 8001;
[dataset_8s,~] = crop_center_jitter_trials(datasetFullTrials,centerIndices, ...
    alignedSamples,0,0,jitterSeed);
[dataset3_jittered,shift_used_samples] = crop_center_jitter_trials( ...
    datasetFullTrials,centerIndices,windowSamples,sigmaSamples,clipSamples,jitterSeed);
labels_table = build_labels_table(shapeDataSet,dataset3_jittered);

num = struct('shape',numClasses,'trial',numTrials,'etr',1,'group',1);
[featurepool,~,featureMeta] = build_feature_pool( ...
    dataset3_jittered,1000,num);
features_with_labels = attach_feature_labels(featurepool,labels_table);

save(fullfile(outputRoot,'aligned_8s.mat'),'dataset_8s','centerIndices','-v7.3');
save(fullfile(outputRoot,'dataset2.mat'),'dataset3_jittered', ...
    'shift_used_samples','centerIndices','-v7.3');
save(fullfile(outputRoot,'labels_table.mat'),'labels_table');
save(fullfile(outputRoot,'features_with_labels.mat'), ...
    'features_with_labels','featureMeta','-v7.3');
fprintf('[Prepared data] %d classes x %d trials written to %s\n', ...
    numClasses,numTrials,outputRoot);

function value = parse_integer(name,fallback)
text = getenv(name);
if isempty(text), value=fallback; else, value=str2double(text); end
assert(isfinite(value) && value==round(value), '%s must be an integer.',name);
end
