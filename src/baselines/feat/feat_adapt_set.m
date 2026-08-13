function adapted = feat_adapt_set(adapterNet, setEmbedding, training)
% Apply FEAT set-to-set attention to a D x S embedding set.

sequence = dlarray(stripdims(setEmbedding), 'CTB');
if training
    adapted = forward(adapterNet, sequence, 'Outputs','adapted_set');
else
    adapted = predict(adapterNet, sequence, 'Outputs','adapted_set');
end
adapted = reshape(stripdims(adapted), size(setEmbedding,1), size(setEmbedding,2));
end
