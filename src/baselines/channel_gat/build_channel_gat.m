function net = build_channel_gat(cfg, numClasses)
% Complete-graph attention over four tactile sensor-channel nodes.

d = cfg.gat.embeddingDim;
h = cfg.gat.numHeads;
drop = cfg.gat.dropout;
inputDim = 2*cfg.gat.temporalBins + 6;

graph = layerGraph();
stem = [
    sequenceInputLayer(inputDim, 'Normalization','none', 'MinLength',4, 'Name','nodes')
    fullyConnectedLayer(d, 'Name','node_projection')
    geluLayer('Name','node_gelu')
    layerNormalizationLayer('Name','node_features')];
attention = selfAttentionLayer(h, d, 'OutputSize',d, ...
    'DropoutProbability',drop, 'Name','graph_attention');
addAttention = additionLayer(2, 'Name','add_graph_attention');
lnAttention = layerNormalizationLayer('Name','graph_ln');
feedForward = [
    fullyConnectedLayer(2*d, 'Name','graph_ff1')
    geluLayer('Name','graph_ff_gelu')
    dropoutLayer(drop, 'Name','graph_ff_drop')
    fullyConnectedLayer(d, 'Name','graph_ff2')];
addFeedForward = additionLayer(2, 'Name','add_graph_ff');
head = [
    layerNormalizationLayer('Name','graph_encoder_out')
    globalAveragePooling1dLayer('Name','graph_pool')
    fullyConnectedLayer(d, 'Name','embed')
    layerNormalizationLayer('Name','embed_ln')
    dropoutLayer(drop, 'Name','head_drop')
    fullyConnectedLayer(numClasses, 'Name','classifier')
    softmaxLayer('Name','probabilities')];

graph = addLayers(graph, stem);
graph = addLayers(graph, attention);
graph = addLayers(graph, addAttention);
graph = addLayers(graph, lnAttention);
graph = addLayers(graph, feedForward);
graph = addLayers(graph, addFeedForward);
graph = addLayers(graph, head);

graph = connectLayers(graph, 'node_features', 'graph_attention');
graph = connectLayers(graph, 'node_features', 'add_graph_attention/in1');
graph = connectLayers(graph, 'graph_attention', 'add_graph_attention/in2');
graph = connectLayers(graph, 'add_graph_attention', 'graph_ln');
graph = connectLayers(graph, 'graph_ln', 'graph_ff1');
graph = connectLayers(graph, 'graph_ln', 'add_graph_ff/in1');
graph = connectLayers(graph, 'graph_ff2', 'add_graph_ff/in2');
graph = connectLayers(graph, 'add_graph_ff', 'graph_encoder_out');

net = dlnetwork(graph);
end
