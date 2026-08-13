function embedding = mamlpp_encoder(params, input, strides)
% Raw four-channel strided CNN used by the MAML++-style baseline.

hidden = dlconv(input, params.conv1W, params.conv1B, ...
    'Stride',[1 strides(1)], 'Padding','same');
hidden = relu(hidden);
hidden = dlconv(hidden, params.conv2W, params.conv2B, ...
    'Stride',[1 strides(2)], 'Padding','same');
hidden = relu(hidden);
hidden = dlconv(hidden, params.conv3W, params.conv3B, ...
    'Stride',[1 strides(3)], 'Padding','same');
hidden = relu(hidden);
pooled = mean(hidden, [1 2]);
pooled = reshape(stripdims(pooled), size(params.fcW,2), []);
embedding = params.fcW * pooled + params.fcB;
end
