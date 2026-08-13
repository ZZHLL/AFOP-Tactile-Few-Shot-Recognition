function sequences = raw_sequence_cells_from_indices(raw, features, indices, stats)
% Return normalized T x C sequences in the cell format expected by trainnet.

indices = indices(:);
sequences = cell(numel(indices),1);
for i = 1:numel(indices)
    row = indices(i);
    x = single(raw{features.y_class(row), features.y_trial(row)});
    sequences{i} = normalize_raw_tensor(x, stats)';
end
end
