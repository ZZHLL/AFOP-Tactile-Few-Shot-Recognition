function net = build_tactile_transformer(cfg, numClasses)
% Temporal Transformer adapted to four-channel tactile sequences.

d = cfg.transformer.embeddingDim;
h = cfg.transformer.numHeads;
drop = cfg.transformer.dropout;

graph = layerGraph();
stem = [
    sequenceInputLayer(4, 'Normalization','none', 'MinLength',cfg.data.signalLength, 'Name','input')
    convolution1dLayer(31, d, 'Stride',4, 'Padding','same', 'Name','conv1')
    layerNormalizationLayer('Name','stem_ln1')
    geluLayer('Name','stem_gelu1')
    convolution1dLayer(15, d, 'Stride',4, 'Padding','same', 'Name','conv2')
    layerNormalizationLayer('Name','stem_ln2')
    geluLayer('Name','stem_out')];
graph = addLayers(graph, stem);

position = sinusoidalPositionEncodingLayer(d, 'Name','position');
addPosition = additionLayer(2, 'Name','add_position');
attention = selfAttentionLayer(h, d, 'OutputSize',d, ...
    'DropoutProbability',drop, 'Name','attention');
addAttention = additionLayer(2, 'Name','add_attention');
lnAttention = layerNormalizationLayer('Name','attention_ln');
feedForward = [
    convolution1dLayer(1, 2*d, 'Name','ff1')
    geluLayer('Name','ff_gelu')
    dropoutLayer(drop, 'Name','ff_drop')
    convolution1dLayer(1, d, 'Name','ff2')];
addFeedForward = additionLayer(2, 'Name','add_ff');
head = [
    layerNormalizationLayer('Name','encoder_out')
    globalAveragePooling1dLayer('Name','pool')
    fullyConnectedLayer(d, 'Name','embed')
    layerNormalizationLayer('Name','embed_ln')
    dropoutLayer(drop, 'Name','head_drop')
    fullyConnectedLayer(numClasses, 'Name','classifier')
    softmaxLayer('Name','probabilities')];

graph = addLayers(graph, position);
graph = addLayers(graph, addPosition);
graph = addLayers(graph, attention);
graph = addLayers(graph, addAttention);
graph = addLayers(graph, lnAttention);
graph = addLayers(graph, feedForward);
graph = addLayers(graph, addFeedForward);
graph = addLayers(graph, head);

graph = connectLayers(graph, 'stem_out', 'position');
graph = connectLayers(graph, 'stem_out', 'add_position/in1');
graph = connectLayers(graph, 'position', 'add_position/in2');
graph = connectLayers(graph, 'add_position', 'attention');
graph = connectLayers(graph, 'add_position', 'add_attention/in1');
graph = connectLayers(graph, 'attention', 'add_attention/in2');
graph = connectLayers(graph, 'add_attention', 'attention_ln');
graph = connectLayers(graph, 'attention_ln', 'ff1');
graph = connectLayers(graph, 'attention_ln', 'add_ff/in1');
graph = connectLayers(graph, 'ff2', 'add_ff/in2');
graph = connectLayers(graph, 'add_ff', 'encoder_out');

net = dlnetwork(graph);
end
