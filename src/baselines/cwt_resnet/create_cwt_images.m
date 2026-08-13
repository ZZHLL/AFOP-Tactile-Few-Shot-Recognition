function create_cwt_images(TactileData, imageRoot, targetImageSize, parameters)
% Convert each four-channel raw trial into one vertically tiled CWT image.

[N_shape, N_trial] = size(TactileData);
N_SE = size(TactileData{1,1},1);
wavelet_type = parameters.waveletType;
voicesPO = parameters.voicesPerOctave;

kk = 1;
for i = 1:N_shape
    for j = 1:N_trial
    im = cell(1, N_SE);
    signalLength = size(TactileData{i,j},2);
    fb = cwtfilterbank('SignalLength',signalLength, "Wavelet", wavelet_type,'VoicesPerOctave',voicesPO);
    for jj = 1:N_SE
        data = TactileData{i,j}(jj,:);
        cfs = abs(fb.wt(data));
        im{jj} = ind2rgb(im2uint8(rescale(cfs)),jet(128));        
    end
    im_merged = imtile(im, 'GridSize', [N_SE 1]);
    labels = i;

    imgLoc = fullfile(imageRoot,['S',num2str(labels)]);
    imFileName = strcat(num2str(kk),'.jpg');
    if ~exist (imgLoc, 'dir')
        mkdir(imgLoc);
    end
    imwrite(imresize(im_merged,targetImageSize),fullfile(imgLoc,imFileName));
    kk = kk+1;
    end
end

end

