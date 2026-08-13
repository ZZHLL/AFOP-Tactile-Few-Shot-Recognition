function [segmentedData,segmentIndices,archetype,cleanedData,outlierIndices] = ...
    autoSegmentSignal(signalData,channelToAnalyze,numTrials,baselineRatio, ...
    enablePlotting,outlierConfig,shapeName,plotAllChannels)
% Automatically segment periodic four-channel tactile recordings.
%
% Set joint_search_enabled to scan both onset and trial length.

arguments
    signalData double
    channelToAnalyze (1,1) double {mustBeInteger,mustBePositive}
    numTrials (1,1) double {mustBeInteger,mustBePositive}
    baselineRatio (1,1) double {mustBePositive,mustBeLessThan(baselineRatio,1)}
    enablePlotting (1,1) logical = false
    outlierConfig struct = struct()
    shapeName = ""
    plotAllChannels (1,1) logical = false
end

fs = 1000;
assert(channelToAnalyze <= size(signalData,1),'Invalid analysis channel.');
assert(size(signalData,2) > 30*fs,'The recording is too short for segmentation.');
outlierConfig = apply_defaults(outlierConfig);
signal = signalData(channelToAnalyze,:);

fprintf('[AutoSeg] %s: estimating the trial period.\n',string(shapeName));
zeroMean = signal-mean(signal,'omitnan');
[values,lags] = xcorr(zeroMean,'coeff');
[peaks,locations] = findpeaks(values,'MinPeakHeight',0.05, ...
    'MinPeakDistance',5*fs);
positive = lags(locations) >= 15*fs & lags(locations) <= 30*fs;
assert(any(positive), ...
    'Autocorrelation found no period in the expected 15-30 s range.');
validPeaks = peaks(positive);
validLocations = locations(positive);
[~,bestPeak] = max(validPeaks);
approximatePeriod = lags(validLocations(bestPeak));

gradient = diff(signal);
if outlierConfig.joint_search_enabled
    % Center candidate onsets and lengths on the estimated landmark and period.
    landmarkRange = 2:min(approximatePeriod-1,numel(gradient));
    assert(~isempty(landmarkRange),'The first-period gradient range is empty.');
    [~,relativeLandmark] = max(-gradient(landmarkRange));
    gradientLandmark = landmarkRange(relativeLandmark);

    onsetHalfCount = floor(outlierConfig.onset_range_ratio*approximatePeriod / ...
        outlierConfig.onset_step_samples);
    candidateStarts = gradientLandmark + ...
        (-onsetHalfCount:onsetHalfCount)*outlierConfig.onset_step_samples;
    candidateStarts = unique(candidateStarts(candidateStarts >= 1));

    lengthHalfCount = floor(outlierConfig.length_range_ratio*approximatePeriod / ...
        outlierConfig.length_step_samples);
    candidateLengths = approximatePeriod + ...
        (-lengthHalfCount:lengthHalfCount)*outlierConfig.length_step_samples;
    candidateLengths = unique(candidateLengths(candidateLengths > 0));
    fprintf(['[AutoSeg] period %.3f s, joint onset/length search: ' ...
        '%d x %d candidates around v*=%d.\n'],approximatePeriod/fs, ...
        numel(candidateStarts),numel(candidateLengths),gradientLandmark);
else
    [~,valleys] = findpeaks(-gradient, ...
        'MinPeakDistance',outlierConfig.min_gap_ratio*approximatePeriod);
    assert(numel(valleys) >= 2, ...
        'Template calibration found fewer than two reliable cycle landmarks.');
    templateStart = valleys(2)+round(0.15*approximatePeriod);
    candidateStarts = templateStart;
    candidateLengths = floor(0.90*approximatePeriod):20: ...
        floor(1.10*approximatePeriod);
    fprintf('[AutoSeg] period %.3f s, fixed template start %d.\n', ...
        approximatePeriod/fs,templateStart);
end

[startGrid,lengthGrid] = ndgrid(candidateStarts,candidateLengths);
candidateStarts = startGrid(:);
candidateLengths = lengthGrid(:);
solutions = repmat(struct('score',-inf,'start',0,'length',0, ...
    'locations',[]),numel(candidateLengths),1);
minGap = outlierConfig.min_gap_ratio*approximatePeriod;
maxGap = outlierConfig.max_gap_ratio*approximatePeriod;
peakMinHeight = outlierConfig.peak_min_height;
jointSearchEnabled = outlierConfig.joint_search_enabled;
parfor i = 1:numel(candidateLengths)
    templateStart = candidateStarts(i);
    trialLength = candidateLengths(i);
    templateEnd = templateStart+trialLength-1;
    if templateEnd > numel(signal), continue; end
    template = signal(templateStart:templateEnd);
    correlation = normxcorr2(template,signal);
    correlation = correlation(trialLength:end-(trialLength-1));
    [scores,starts] = findpeaks(correlation, ...
        'MinPeakHeight',peakMinHeight, ...
        'MinPeakDistance',minGap);
    if numel(starts) < numTrials, continue; end
    if jointSearchEnabled
        [starts,selectedScores] = select_regular_peak_sequence( ...
            starts,scores,numTrials,minGap,maxGap);
        if isempty(starts), continue; end
    else
        [scores,order] = sort(scores,'descend');
        selectedScores = scores(1:numTrials);
        starts = sort(starts(order(1:numTrials)));
        if any(diff(starts) > maxGap), continue; end
    end
    solutions(i).score = mean(selectedScores);
    solutions(i).start = templateStart;
    solutions(i).length = trialLength;
    solutions(i).locations = starts;
end

[bestScore,bestIndex] = max([solutions.score]);
assert(isfinite(bestScore),'No valid segmentation solution was found.');
bestLength = solutions(bestIndex).length;
starts = solutions(bestIndex).locations(:);
fprintf('[AutoSeg] selected onset %d, length %d, score %.4f.\n', ...
    solutions(bestIndex).start,bestLength,bestScore);
segmentIndices = [starts,starts+bestLength-1];
assert(all(segmentIndices(:,2) <= size(signalData,2)), ...
    'A selected segment extends beyond the recording.');

numChannels = size(signalData,1);
segmentedData = zeros(bestLength,numChannels,numTrials,'like',signalData);
baselinePoints = max(1,round(baselineRatio*bestLength));
for i = 1:numTrials
    segment = signalData(:,segmentIndices(i,1):segmentIndices(i,2));
    baseline = mean(segment(:,1:baselinePoints),2,'omitnan');
    segmentedData(:,:,i) = (segment-baseline)';
end
archetype = mean(segmentedData(:,channelToAnalyze,:),3,'omitnan');

outlierIndices = [];
if outlierConfig.enabled
    trialRms = squeeze(rms(segmentedData(:,channelToAnalyze,:),1));
    spread = std(trialRms);
    if spread > 0
        z = abs((trialRms-mean(trialRms))/spread);
        outlierIndices = find(z > outlierConfig.z_threshold)';
    end
end
cleanedData = segmentedData(:,:,setdiff(1:numTrials,outlierIndices));

if enablePlotting
    channels = channelToAnalyze;
    if plotAllChannels, channels = 1:numChannels; end
    for channel = channels
        plot_segmentation(signalData,segmentedData,segmentIndices, ...
            outlierIndices,channel,shapeName);
    end
end
fprintf('[AutoSeg] %s: %d trials, length %d, %d outliers.\n', ...
    string(shapeName),numTrials,bestLength,numel(outlierIndices));
end

function config = apply_defaults(config)
if ~isfield(config,'enabled'), config.enabled = false; end
if ~isfield(config,'z_threshold'), config.z_threshold = 2.5; end
if ~isfield(config,'joint_search_enabled'), config.joint_search_enabled = false; end
if ~isfield(config,'onset_range_ratio'), config.onset_range_ratio = 0.10; end
if ~isfield(config,'length_range_ratio'), config.length_range_ratio = 0.10; end
if ~isfield(config,'onset_step_samples'), config.onset_step_samples = 20; end
if ~isfield(config,'length_step_samples'), config.length_step_samples = 20; end
if ~isfield(config,'peak_min_height'), config.peak_min_height = 0.7; end
if ~isfield(config,'min_gap_ratio'), config.min_gap_ratio = 0.8; end
if ~isfield(config,'max_gap_ratio'), config.max_gap_ratio = 1.2; end
validateattributes(config.joint_search_enabled,{'logical','numeric'}, ...
    {'scalar'},mfilename,'joint_search_enabled');
validateattributes(config.onset_range_ratio,{'numeric'}, ...
    {'scalar','real','finite','nonnegative'},mfilename,'onset_range_ratio');
validateattributes(config.length_range_ratio,{'numeric'}, ...
    {'scalar','real','finite','nonnegative'},mfilename,'length_range_ratio');
validateattributes(config.onset_step_samples,{'numeric'}, ...
    {'scalar','real','finite','integer','positive'},mfilename,'onset_step_samples');
validateattributes(config.length_step_samples,{'numeric'}, ...
    {'scalar','real','finite','integer','positive'},mfilename,'length_step_samples');
validateattributes(config.peak_min_height,{'numeric'}, ...
    {'scalar','real','finite','>=',-1,'<=',1},mfilename,'peak_min_height');
validateattributes(config.min_gap_ratio,{'numeric'}, ...
    {'scalar','real','finite','positive'},mfilename,'min_gap_ratio');
validateattributes(config.max_gap_ratio,{'numeric'}, ...
    {'scalar','real','finite','>',config.min_gap_ratio},mfilename,'max_gap_ratio');
config.joint_search_enabled = logical(config.joint_search_enabled);
end

function [selectedStarts,selectedScores] = select_regular_peak_sequence( ...
    starts,scores,numTrials,minGap,maxGap)
starts = starts(:);
scores = scores(:);
[starts,order] = sort(starts);
scores = scores(order);
numPeaks = numel(starts);

bestSum = -inf(numPeaks,numTrials);
previous = zeros(numPeaks,numTrials,'uint32');
bestSum(:,1) = scores;
for count = 2:numTrials
    for current = 1:numPeaks
        gaps = starts(current)-starts(1:current-1);
        validPrevious = find(gaps >= minGap & gaps <= maxGap);
        if isempty(validPrevious), continue; end
        [predecessorSum,localIndex] = max(bestSum(validPrevious,count-1));
        if isfinite(predecessorSum)
            predecessor = validPrevious(localIndex);
            bestSum(current,count) = predecessorSum+scores(current);
            previous(current,count) = uint32(predecessor);
        end
    end
end

[totalScore,last] = max(bestSum(:,numTrials));
if ~isfinite(totalScore)
    selectedStarts = [];
    selectedScores = [];
    return
end
indices = zeros(numTrials,1);
indices(end) = last;
for count = numTrials:-1:2
    indices(count-1) = double(previous(indices(count),count));
end
selectedStarts = starts(indices)';
selectedScores = scores(indices)';
end

function plot_segmentation(raw,segments,indices,outliers,channel,shapeName)
figure('Name',sprintf('%s - channel %d',string(shapeName),channel), ...
    'Position',[100 100 1200 800]);
subplot(2,1,1); plot(raw(channel,:),'Color',[0.75 0.75 0.75]); hold on;
for i = 1:size(indices,1)
    style = '-'; if ismember(i,outliers), style = '-.'; end
    range = indices(i,1):indices(i,2);
    plot(range,raw(channel,range),style,'LineWidth',1);
end
grid on; axis tight; xlabel('Sample'); ylabel('Voltage (V)');
title(sprintf('%s: detected trials',string(shapeName)));
subplot(2,1,2); plot(squeeze(segments(:,channel,:)),':'); hold on;
plot(mean(segments(:,channel,:),3,'omitnan'),'r','LineWidth',2);
grid on; axis tight; xlabel('Sample within trial'); ylabel('Corrected voltage (V)');
title('Baseline-corrected trials and archetype');
end
