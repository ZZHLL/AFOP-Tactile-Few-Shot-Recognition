function logits = feat_euclidean_logits(samples, centers, temperature)
% Negative squared Euclidean distance divided by the FEAT temperature.

samples = stripdims(samples);
centers = stripdims(centers);
sampleSq = sum(samples.^2, 1);
centerSq = sum(centers.^2, 1)';
distance = centerSq + sampleSq - 2 * (centers' * samples);
logits = -distance / temperature;
end
