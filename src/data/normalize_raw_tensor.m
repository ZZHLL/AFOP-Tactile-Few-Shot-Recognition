function X = normalize_raw_tensor(X, stats)
% Apply source-derived channel normalization to C x T x B data.

mu = reshape(single(stats.mean), [], 1, 1);
sigma = reshape(single(stats.std), [], 1, 1);
X = (single(X) - mu) ./ max(sigma, single(1e-6));
end

