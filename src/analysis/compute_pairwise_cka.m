function tableOut = compute_pairwise_cka(representations)
% Compute pairwise linear CKA for fields of a representation structure.

names = string(fieldnames(representations));
matrix = zeros(numel(names),numel(names));
for row = 1:numel(names)
    for column = row:numel(names)
        value = linear_cka(representations.(names(row)), ...
            representations.(names(column)));
        matrix(row,column)=value; matrix(column,row)=value;
    end
end
tableOut = array2table(matrix,'VariableNames',matlab.lang.makeValidName(names), ...
    'RowNames',cellstr(names));
end
