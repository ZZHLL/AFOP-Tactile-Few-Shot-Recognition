function frontend = fit_feature_frontend(method,features,fitRows,options)
% Fit one source-only ranking or projection frontend.

arguments
    method (1,1) string
    features struct
    fitRows double
    options.ReliefNeighbors (1,1) double = 10
end
X = features.X_all;
y = features.y_class(:);
fitRows = fitRows(:);
[scaledT,minmaxSettings] = mapminmax(X(fitRows,:)',0,1);
scaled = scaledT';

frontend = struct('method',lower(method),'fitRows',fitRows, ...
    'minmaxSettings',minmaxSettings);
switch lower(method)
    case "nca"
        model = fscnca(scaled,y(fitRows),'Verbose',0);
        weights = model.FeatureWeights(:);
        [~,rank] = sort(weights,'descend');
        frontend.kind = "ranking";
        frontend.rank = rank;
        frontend.weights = weights;
    case "relieff"
        neighbors = min(options.ReliefNeighbors,size(scaled,1)-1);
        [rank,weights] = relieff(scaled,y(fitRows),neighbors);
        frontend.kind = "ranking";
        frontend.rank = rank(:);
        frontend.weights = weights(:);
        frontend.neighbors = neighbors;
    case "mrmr"
        rank = fscmrmr(scaled,y(fitRows));
        rank = rank(:);
        if numel(rank)<size(scaled,2)
            rank = [rank;setdiff((1:size(scaled,2))',rank,'stable')];
        end
        frontend.kind = "ranking";
        frontend.rank = rank;
        frontend.weights = [];
    case "pca"
        [coeff,~,latent,~,explained,mu] = pca(scaled,'Centered',true);
        frontend.kind = "projection";
        frontend.coeff = coeff;
        frontend.mu = mu;
        frontend.latent = latent;
        frontend.explained = explained;
    otherwise
        error('Unknown feature frontend: %s',method);
end
end
