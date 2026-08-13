function representation = apply_feature_frontend(frontend,X)
% Apply a fitted ranking or PCA frontend without refitting target data.

switch frontend.kind
    case "ranking"
        representation = X(:,frontend.rank);
    case "projection"
        scaled = mapminmax('apply',X',frontend.minmaxSettings)';
        representation = (scaled-frontend.mu)*frontend.coeff;
    otherwise
        error('Unsupported frontend kind: %s',frontend.kind);
end
end
