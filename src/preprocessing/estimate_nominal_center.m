function centerIndex = estimate_nominal_center(archetype,referenceChannel,roi)
% Estimate the event center between the steepest falling and rising slopes.
% archetype is samples-by-channels and roi is [firstSample lastSample].

arguments
    archetype double
    referenceChannel (1,1) double {mustBeInteger,mustBePositive}
    roi (1,2) double {mustBeInteger,mustBePositive}
end
assert(referenceChannel<=size(archetype,2),'Invalid reference channel.');
assert(roi(1)<roi(2) && roi(2)<=size(archetype,1),'ROI is outside the archetype.');
gradient = diff(archetype(roi(1):roi(2),referenceChannel));
[~,falling] = min(gradient); [~,rising] = max(gradient);
centerIndex = roi(1)+round((falling+rising)/2)-1;
end
