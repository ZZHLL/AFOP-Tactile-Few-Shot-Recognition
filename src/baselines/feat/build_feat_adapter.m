function adapterNet = build_feat_adapter(cfg)
% Build the one-head residual self-attention adapter.

d = cfg.featOfficial.embeddingDim;
graph = layerGraph();
input = sequenceInputLayer(d, 'Normalization','none', 'Name','set_input');
attention = selfAttentionLayer(1, d, 'OutputSize',d, ...
    'DropoutProbability',cfg.featOfficial.attentionScoreDropout, ...
    'Name','single_head_attention');
outputDropout = dropoutLayer(cfg.featOfficial.attentionOutputDropout, ...
    'Name','attention_output_dropout');
residual = additionLayer(2, 'Name','attention_residual');
output = layerNormalizationLayer('Name','adapted_set');
graph = addLayers(graph, input);
graph = addLayers(graph, attention);
graph = addLayers(graph, outputDropout);
graph = addLayers(graph, residual);
graph = addLayers(graph, output);
graph = connectLayers(graph, 'set_input', 'single_head_attention');
graph = connectLayers(graph, 'set_input', 'attention_residual/in1');
graph = connectLayers(graph, 'single_head_attention', 'attention_output_dropout');
graph = connectLayers(graph, 'attention_output_dropout', 'attention_residual/in2');
graph = connectLayers(graph, 'attention_residual', 'adapted_set');
adapterNet = dlnetwork(graph);
end
