clear; clc;

workspaceRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
sourcePath = getenv_default('AFOP_ALIGNED8S_SOURCE', ...
    fullfile(workspaceRoot, 'data', 'aligned_8s.mat'));
outputPath = getenv_default('AFOP_PHYSICAL_TARGET_OUTPUT', ...
    fullfile(workspaceRoot, 'artifacts', 'physical_target_sigma200.mat'));

assert(isfile(sourcePath), 'Missing aligned 8-s source: %s', sourcePath);
if isfile(outputPath)
    fprintf('[physical target] existing artifact retained: %s\n', outputPath);
    return;
end
source = load(sourcePath, 'dataset_8s');
assert(isequal(size(source.dataset_8s), [36 60]), 'dataset_8s must be 36 x 60.');

aug = struct('seed', 20250814, 'samplingRate', 1000, 'T_out', 4000, ...
    'shiftSigma', 200, 'shiftClip', 600, 'ampRange', [0.7 1.3], ...
    'speedRange', [0.85 1.15], 'noiseRel', 0.05);
rng(aug.seed, 'twister');

dataset_aug = cell(36, 60);
info = struct();
info.shift_used_samples = zeros(36, 60);
info.amp_used = ones(36, 60);
info.speed_used = ones(36, 60);
for classId = 1:36
    for trialId = 1:60
        X8 = source.dataset_8s{classId, trialId};
        assert(size(X8,1) == 4 && size(X8,2) >= 8000, ...
            'Invalid aligned 8-s trial at class %d trial %d.', classId, trialId);
        shift = round(aug.shiftSigma * randn());
        shift = max(-aug.shiftClip, min(aug.shiftClip, shift));
        startIndex = 2001 + shift;
        X0 = double(X8(:, startIndex:startIndex + aug.T_out - 1));

        amplitude = aug.ampRange(1) + diff(aug.ampRange)*rand();
        speed = aug.speedRange(1) + diff(aug.speedRange)*rand();
        Y = time_warp_to_length(X0*amplitude, speed, aug.T_out);
        channelStd = std(Y, 0, 2);
        Y = Y + (aug.noiseRel*channelStd).*randn(size(Y));

        dataset_aug{classId, trialId} = single(Y);
        info.shift_used_samples(classId, trialId) = shift;
        info.amp_used(classId, trialId) = amplitude;
        info.speed_used(classId, trialId) = speed;
    end
end

info.aug = aug;
metadata = struct();
metadata.protocol = "strength/speed/noise target with default center jitter";
metadata.source = string(sourcePath);
metadata.centerConvention = "aligned 8-s trial, 4000-sample crop centered at index 4001";
metadata.shift = "Gaussian sigma=200 samples, clipped to +/-600 samples";
metadata.amplitudeRange = aug.ampRange;
metadata.speedRange = aug.speedRange;
metadata.noiseRelativeStd = aug.noiseRel;
metadata.seed = aug.seed;
metadata.created = string(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));

for classId = 1:36
    for trialId = 1:60
        X = dataset_aug{classId,trialId};
        assert(isequal(size(X),[4 4000]) && all(isfinite(X(:))), ...
            'Invalid generated target trial at class %d trial %d.',classId,trialId);
    end
end
parent = fileparts(outputPath);
if ~exist(parent,'dir'), mkdir(parent); end
save(outputPath,'dataset_aug','info','metadata','-v7.3');
fprintf('[physical target] saved %s\n',outputPath);
fprintf('[physical target] shift mean/std/range = %.2f / %.2f / [%d,%d] samples\n', ...
    mean(info.shift_used_samples(:)),std(info.shift_used_samples(:)), ...
    min(info.shift_used_samples(:)),max(info.shift_used_samples(:)));

function Y = time_warp_to_length(X, speed, outputLength)
[channels,inputLength] = size(X);
warpedLength = max(round(inputLength/speed),2);
oldAxis = 1:inputLength;
newAxis = linspace(1,inputLength,warpedLength);
temporary = zeros(channels,warpedLength);
for channel = 1:channels
    temporary(channel,:) = interp1(oldAxis,X(channel,:),newAxis,'linear','extrap');
end
if warpedLength >= outputLength
    offset = floor((warpedLength-outputLength)/2);
    Y = temporary(:,offset+1:offset+outputLength);
else
    missing = outputLength-warpedLength;
    before = floor(missing/2);
    after = missing-before;
    Y = [repmat(temporary(:,1),1,before),temporary,repmat(temporary(:,end),1,after)];
end
end

function value = getenv_default(name,fallback)
value = getenv(name); if isempty(value), value = fallback; end
end
