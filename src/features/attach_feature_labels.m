function features_with_labels = attach_feature_labels(featurepool, labels_table)
% Align the feature-pool rows with labels_table.

N = size(featurepool,1);
assert(height(labels_table) == N, 'labels_table 行数与 featurepool 行数不一致');

[~, order_colMajor] = sort(labels_table.linear_idx_colMajor, 'ascend');
lt = labels_table(order_colMajor, :);

assert(all(lt.linear_idx_colMajor == (1:N)'), 'col-major 排序后序号仍不连续，检查 labels_table');

features_with_labels = struct();
features_with_labels.X_all      = featurepool;
features_with_labels.y_class    = lt.class_id;
features_with_labels.y_shape12  = lt.shape12_id;
features_with_labels.y_material = lt.material_id;
features_with_labels.y_trial    = lt.trial_id;
features_with_labels.valid      = lt.valid;
features_with_labels.shape_base = lt.shape_base;
features_with_labels.material   = lt.material_std;

bad = ~features_with_labels.valid | any(~isfinite(features_with_labels.X_all),2);
features_with_labels.bad_mask = bad;

fprintf('Constructed labeled feature pool (%d rows x %d features).\n', ...
    N,size(featurepool,2));
end
