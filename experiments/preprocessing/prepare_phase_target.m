clear; clc;

workspaceRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
sourcePath = getenv_default('AFOP_ALIGNED8S_SOURCE', ...
    fullfile(workspaceRoot, 'data', 'aligned_8s.mat'));

shiftSigma = parse_scalar_env('AFOP_SHIFT_SIGMA', 200);
shiftClip = parse_scalar_env('AFOP_SHIFT_CLIP', 3 * shiftSigma);
assert(shiftSigma >= 0 && shiftSigma == round(shiftSigma), ...
    'AFOP_SHIFT_SIGMA must be a nonnegative integer number of samples.');
assert(shiftClip >= 0 && shiftClip == round(shiftClip), ...
    'AFOP_SHIFT_CLIP must be a nonnegative integer number of samples.');

outputPath = getenv_default('AFOP_PHASE_TARGET_OUTPUT', ...
    fullfile(workspaceRoot, 'artifacts', ...
    sprintf('phase_target_sigma%03d.mat', shiftSigma)));

assert(isfile(sourcePath), 'Missing aligned 8-s source: %s', sourcePath);
if isfile(outputPath)
    saved = load(outputPath, 'metadata');
    assert(saved.metadata.shiftSigmaSamples == shiftSigma, ...
        'Existing target sigma does not match the requested sigma.');
    assert(saved.metadata.shiftClipSamples == shiftClip, ...
        'Existing target clip does not match the requested clip.');
    fprintf('[phase target] verified existing artifact: %s\n', outputPath);
    return;
end

source = load(sourcePath, 'dataset_8s');
assert(isequal(size(source.dataset_8s), [36 60]), ...
    'dataset_8s must be 36 x 60.');

aug = struct('seed', 20250814, 'samplingRate', 1000, 'T_out', 4000, ...
    'shiftSigma', shiftSigma, 'shiftClip', shiftClip, ...
    'ampRange', [0.7 1.3], 'speedRange', [0.85 1.15], ...
    'noiseRel', 0.05);
rng(aug.seed, 'twister');

dataset_aug = cell(36, 60);
info = struct();
info.shift_used_samples = zeros(36, 60);
info.amp_used = ones(36, 60);
info.speed_used = ones(36, 60);
for classId = 1:36
    for trialId = 1:60
        X8 = source.dataset_8s{classId, trialId};
        assert(isequal(size(X8), [4 8001]) && all(isfinite(X8(:))), ...
            'Invalid aligned 8-s trial at class %d trial %d.', ...
            classId, trialId);

        shift = round(aug.shiftSigma * randn());
        shift = max(-aug.shiftClip, min(aug.shiftClip, shift));
        startIndex = 2001 + shift;
        stopIndex = startIndex + aug.T_out - 1;
        assert(startIndex >= 1 && stopIndex <= size(X8, 2), ...
            'Crop exceeds the aligned 8-s trial at class %d trial %d.', ...
            classId, trialId);
        X0 = double(X8(:, startIndex:stopIndex));

        amplitude = aug.ampRange(1) + diff(aug.ampRange) * rand();
        speed = aug.speedRange(1) + diff(aug.speedRange) * rand();
        Y = time_warp_to_length(X0 * amplitude, speed, aug.T_out);
        channelStd = std(Y, 0, 2);
        Y = Y + (aug.noiseRel * channelStd) .* randn(size(Y));

        dataset_aug{classId, trialId} = single(Y);
        info.shift_used_samples(classId, trialId) = shift;
        info.amp_used(classId, trialId) = amplitude;
        info.speed_used(classId, trialId) = speed;
    end
end

info.aug = aug;
metadata = struct();
metadata.protocol = "direct center-jitter crop with force/speed/noise perturbation";
metadata.source = string(sourcePath);
metadata.centerConvention = ...
    "aligned 8001-sample trial; 4000 samples start at index 2001 + shift";
metadata.shiftSigmaSamples = shiftSigma;
metadata.shiftClipSamples = shiftClip;
metadata.amplitudeRange = aug.ampRange;
metadata.speedRange = aug.speedRange;
metadata.noiseRelativeStd = aug.noiseRel;
metadata.seed = aug.seed;
metadata.created = string(datetime('now', ...
    'Format', 'yyyy-MM-dd HH:mm:ss'));

for classId = 1:36
    for trialId = 1:60
        X = dataset_aug{classId, trialId};
        assert(isequal(size(X), [4 4000]) && all(isfinite(X(:))), ...
            'Invalid generated target trial at class %d trial %d.', ...
            classId, trialId);
    end
end

parent = fileparts(outputPath);
if ~exist(parent, 'dir'), mkdir(parent); end
save(outputPath, 'dataset_aug', 'info', 'metadata', '-v7.3');
fprintf('[phase target] saved %s\n', outputPath);
fprintf(['[phase target] sigma/clip=%d/%d | shift mean/std/range=' ...
    '%.2f/%.2f/[%d,%d] | clipped=%d\n'], ...
    shiftSigma, shiftClip, mean(info.shift_used_samples(:)), ...
    std(info.shift_used_samples(:)), min(info.shift_used_samples(:)), ...
    max(info.shift_used_samples(:)), ...
    nnz(abs(info.shift_used_samples(:)) == shiftClip));

function Y = time_warp_to_length(X, speed, outputLength)
[channels, inputLength] = size(X);
warpedLength = max(round(inputLength / speed), 2);
oldAxis = 1:inputLength;
newAxis = linspace(1, inputLength, warpedLength);
temporary = zeros(channels, warpedLength);
for channel = 1:channels
    temporary(channel, :) = interp1(oldAxis, X(channel, :), ...
        newAxis, 'linear', 'extrap');
end
if warpedLength >= outputLength
    offset = floor((warpedLength - outputLength) / 2);
    Y = temporary(:, offset + 1:offset + outputLength);
else
    missing = outputLength - warpedLength;
    before = floor(missing / 2);
    after = missing - before;
    Y = [repmat(temporary(:, 1), 1, before), temporary, ...
        repmat(temporary(:, end), 1, after)];
end
end

function value = parse_scalar_env(name, fallback)
textValue = getenv(name);
if isempty(textValue)
    value = fallback;
else
    value = str2double(textValue);
    assert(isfinite(value) && isscalar(value), ...
        'Environment variable %s must be a finite scalar.', name);
end
end

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end
