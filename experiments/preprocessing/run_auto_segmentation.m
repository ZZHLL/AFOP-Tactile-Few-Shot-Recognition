clear; clc;
% Segment all raw TXT recordings in one folder without hard-coded paths.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
originalFolder = pwd;
folderCleanup = onCleanup(@() cd(originalFolder));
cd(root);
addpath(genpath(fullfile(root,'src'))); setup_afop_path();
inputRoot = string(getenv('AFOP_SEGMENT_INPUT'));
assert(strlength(inputRoot)>0 && isfolder(inputRoot), ...
    'Set AFOP_SEGMENT_INPUT to a folder containing raw TXT recordings.');
outputRoot = string(getenv_default('AFOP_SEGMENT_OUTPUT', ...
    fullfile(root,'artifacts','segmentation')));
analysisChannel = str2double(getenv_default('AFOP_SEGMENT_CHANNEL','3'));
numTrials = str2double(getenv_default('AFOP_SEGMENT_TRIALS','60'));
enablePlots = parse_boolean(getenv_default('AFOP_SEGMENT_PLOTS','false'));

files = dir(fullfile(inputRoot,'*.txt'));
assert(~isempty(files),'No TXT recordings found in %s.',inputRoot);
[~,order] = sort(lower(string({files.name})));
files = files(order);
numShapes = numel(files);
dataset0 = cell(1,numShapes);
dataset0_filtered_sg = cell(1,numShapes);
shapeDataSet = repmat(struct('shapeName',"",'sourceFile',"", ...
    'fullData',[],'cleanedData',[],'segmentIndices',[], ...
    'outlierIndices',[],'archetype',[],'status',"pending",'error',""),1,numShapes);

filterOrder = 3;
filterFrameLength = 1001;
outlierConfig = struct( ...
    'enabled',true, ...
    'z_threshold',2.5, ...
    'joint_search_enabled',parse_boolean(getenv_default( ...
        'AFOP_SEGMENT_JOINT_SEARCH','false')), ...
    'onset_range_ratio',str2double(getenv_default( ...
        'AFOP_SEGMENT_ONSET_RANGE','0.10')), ...
    'length_range_ratio',str2double(getenv_default( ...
        'AFOP_SEGMENT_LENGTH_RANGE','0.10')), ...
    'onset_step_samples',str2double(getenv_default( ...
        'AFOP_SEGMENT_ONSET_STEP','20')), ...
    'length_step_samples',str2double(getenv_default( ...
        'AFOP_SEGMENT_LENGTH_STEP','20')), ...
    'peak_min_height',0.7, ...
    'min_gap_ratio',0.8, ...
    'max_gap_ratio',1.2);
for i = 1:numShapes
    path = fullfile(files(i).folder,files(i).name);
    matrix = readmatrix(path);
    matrix = matrix(all(isfinite(matrix),2),:);
    if size(matrix,2) == 5
        channels = matrix(:,2:5)';
    elseif size(matrix,2) == 4
        channels = matrix';
    else
        error('Expected four channels, optionally preceded by time: %s',path);
    end
    assert(size(channels,2)>=filterFrameLength, ...
        'Recording is shorter than the smoothing frame: %s',path);
    dataset0{i} = channels;
    filtered = zeros(size(channels));
    for channel = 1:4
        filtered(channel,:) = sgolayfilt(channels(channel,:), ...
            filterOrder,filterFrameLength);
    end
    dataset0_filtered_sg{i} = filtered;
    [~,name] = fileparts(files(i).name);
    shapeDataSet(i).shapeName = string(name);
    shapeDataSet(i).sourceFile = string(path);
    try
        [shapeDataSet(i).fullData,shapeDataSet(i).segmentIndices, ...
            shapeDataSet(i).archetype,shapeDataSet(i).cleanedData, ...
            shapeDataSet(i).outlierIndices] = autoSegmentSignal(filtered, ...
            analysisChannel,numTrials,0.1,enablePlots,outlierConfig,name,false);
        shapeDataSet(i).status = "ok";
    catch exception
        shapeDataSet(i).status = "failed";
        shapeDataSet(i).error = string(exception.message);
        warning('AFOP:SegmentationFailed','%s: %s',name,exception.message);
    end
end

if ~isfolder(outputRoot), mkdir(outputRoot); end
stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
outputPath = fullfile(outputRoot,['segmentation_' stamp '.mat']);
config = struct('inputRoot',inputRoot,'analysisChannel',analysisChannel, ...
    'numTrials',numTrials,'filterOrder',filterOrder, ...
    'filterFrameLength',filterFrameLength,'outlierConfig',outlierConfig);
save(outputPath,'dataset0','dataset0_filtered_sg','shapeDataSet','config','-v7.3');
fprintf('[AutoSeg] Saved %d recordings to %s\n',numShapes,outputPath);
clear folderCleanup

function value = getenv_default(name,fallback)
value = getenv(name); if isempty(value), value = fallback; end
end
function value = parse_boolean(text)
value = any(strcmpi(string(text),["1","true","yes","on"]));
end
