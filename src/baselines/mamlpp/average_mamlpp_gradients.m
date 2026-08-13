function average = average_mamlpp_gradients(gradientList)
% Average explicit parameter-gradient structs across a meta-batch.

average = gradientList{1};
names = fieldnames(average);
for task = 2:numel(gradientList)
    for i = 1:numel(names)
        average.(names{i}) = average.(names{i}) + gradientList{task}.(names{i});
    end
end
for i = 1:numel(names)
    average.(names{i}) = average.(names{i}) / numel(gradientList);
end
end
