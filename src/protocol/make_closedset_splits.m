function fold = make_closedset_splits(features_with_labels, ratio_train, ratio_inner, seed)
% Create stratified train/test and source-train/source-validation splits.

if nargin < 2 || isempty(ratio_train), ratio_train = 0.5; end
if nargin < 3 || isempty(ratio_inner), ratio_inner = 0.8; end
if nargin < 4 || isempty(seed),        seed        = 20250812; end
rng(seed);

y = features_with_labels.y_class;
classes = unique(y)';

train_idx = [];
test_idx  = [];
for c = classes
    idx_c = find(y==c);
    n  = numel(idx_c);
    nt = floor(ratio_train * n);
    p  = randperm(n);
    train_idx = [train_idx; idx_c(p(1:nt))];      %#ok<AGROW>
    test_idx  = [test_idx;  idx_c(p(nt+1:end))];  %#ok<AGROW>
end

source_train = [];
source_val   = [];
for c = classes
    idx_c = intersect(find(y==c), train_idx);
    n  = numel(idx_c);
    ni = max(1, floor(ratio_inner * n));
    p  = randperm(n);
    source_train = [source_train; idx_c(p(1:ni))];      %#ok<AGROW>
    source_val   = [source_val;   idx_c(p(ni+1:end))];  %#ok<AGROW>
end

fold = struct();
fold.name         = sprintf('closedset_%d_%d', round(ratio_train*100), round(ratio_inner*100));
fold.train_all    = sort(train_idx);
fold.test_all     = sort(test_idx);
fold.source_train = sort(source_train);
fold.source_val   = sort(source_val);
end
