function labelsTable = build_labels_table(shapeDataSet, dataset)
% Build labels aligned with a class-by-trial tactile dataset.

numClasses = size(dataset,1);
numTrials = size(dataset,2);
assert(numel(shapeDataSet) == numClasses, ...
    'shapeDataSet and dataset contain different class counts.');

shapeBase = strings(numClasses,1);
materialRaw = strings(numClasses,1);
materialStd = strings(numClasses,1);
materialId = zeros(numClasses,1);
for classId = 1:numClasses
    name = string(shapeDataSet(classId).shapeName);
    token = regexp(name, '(Resin|Wood|Al|Aluminum|Steel)(?=(?:_?60)?$)', ...
        'match', 'once', 'ignorecase');
    assert(~isempty(token), 'Unrecognized material suffix in class name: %s', name);
    shapeBase(classId) = regexprep(name, ...
        '(Resin|Wood|Al|Aluminum|Steel)(?:_?60)?$', '', 'ignorecase');
    shapeBase(classId) = regexprep(shapeBase(classId), ...
        '^(?:slid|slide)_data_', '', 'ignorecase');
    shapeBase(classId) = regexprep(shapeBase(classId),'^_+|_+$','');
    materialRaw(classId) = string(token);
    switch lower(string(token))
        case "resin"
            materialId(classId) = 1; materialStd(classId) = "Resin";
        case {"al","aluminum","steel"}
            materialId(classId) = 2; materialStd(classId) = "Metal";
        case "wood"
            materialId(classId) = 3; materialStd(classId) = "Wood";
    end
end

[uniqueShapes,~,shapeIdPerClass] = unique(shapeBase,'stable');
assert(numel(uniqueShapes) * numel(unique(materialId)) == numClasses, ...
    'Expected a complete shape-by-material benchmark.');

numRows = numClasses*numTrials;
class_id = zeros(numRows,1);
shape12_id = zeros(numRows,1);
trial_id = zeros(numRows,1);
row_idx_dataset3 = zeros(numRows,1);
col_idx_dataset3 = zeros(numRows,1);
linear_idx_rowMajor = zeros(numRows,1);
linear_idx_colMajor = zeros(numRows,1);
shape_base = strings(numRows,1);
material_raw = strings(numRows,1);
material_std = strings(numRows,1);
material_id = zeros(numRows,1);
valid = false(numRows,1);

row = 0;
for classId = 1:numClasses
    for trialId = 1:numTrials
        row = row+1;
        class_id(row) = classId;
        shape12_id(row) = shapeIdPerClass(classId);
        trial_id(row) = trialId;
        row_idx_dataset3(row) = classId;
        col_idx_dataset3(row) = trialId;
        linear_idx_rowMajor(row) = (classId-1)*numTrials+trialId;
        linear_idx_colMajor(row) = (trialId-1)*numClasses+classId;
        shape_base(row) = shapeBase(classId);
        material_raw(row) = materialRaw(classId);
        material_std(row) = materialStd(classId);
        material_id(row) = materialId(classId);
        valid(row) = ~isempty(dataset{classId,trialId});
    end
end

labelsTable = table(class_id,shape12_id,trial_id,row_idx_dataset3, ...
    col_idx_dataset3,linear_idx_rowMajor,linear_idx_colMajor,shape_base, ...
    material_raw,material_std,material_id,valid);
end
