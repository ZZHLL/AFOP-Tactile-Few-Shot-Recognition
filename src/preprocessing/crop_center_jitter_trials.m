function [datasetJittered, shifts] = crop_center_jitter_trials( ...
    fullTrials, centerIndices, windowSamples, sigmaSamples, clipSamples, seed)
% Crop naturally shifted windows from segmented trials without circular wrap.
%
% fullTrials    : C-by-T cell array; each cell is channels-by-samples.
% centerIndices : C-by-1 or C-by-T nominal center indices within each trial.

arguments
    fullTrials cell
    centerIndices double
    windowSamples (1,1) double {mustBeInteger,mustBePositive}
    sigmaSamples (1,1) double {mustBeNonnegative}
    clipSamples (1,1) double {mustBeNonnegative}
    seed (1,1) double {mustBeInteger} = 1
end

numClasses = size(fullTrials,1);
numTrials = size(fullTrials,2);
assert(size(centerIndices,1) == numClasses, 'Center count does not match classes.');
assert(size(centerIndices,2) == 1 || size(centerIndices,2) == numTrials, ...
    'centerIndices must be C-by-1 or C-by-T.');

stream = RandStream('mt19937ar','Seed',seed);
datasetJittered = cell(size(fullTrials));
shifts = zeros(size(fullTrials));
left = floor(windowSamples/2);
right = windowSamples-left-1;

for c = 1:numClasses
    for t = 1:numTrials
        x = fullTrials{c,t};
        assert(~isempty(x) && ismatrix(x), 'Empty trial at {%d,%d}.',c,t);
        if size(centerIndices,2) == 1
            nominalCenter = round(centerIndices(c));
        else
            nominalCenter = round(centerIndices(c,t));
        end
        minCenter = 1+left;
        maxCenter = size(x,2)-right;
        assert(maxCenter >= minCenter, 'Trial {%d,%d} is shorter than the crop.',c,t);

        shift = round(sigmaSamples*randn(stream));
        shift = max(-clipSamples,min(clipSamples,shift));
        center = max(minCenter,min(maxCenter,nominalCenter+shift));
        shift = center-nominalCenter;
        startIndex = center-left;
        datasetJittered{c,t} = x(:,startIndex:startIndex+windowSamples-1);
        shifts(c,t) = shift;
    end
end
end
