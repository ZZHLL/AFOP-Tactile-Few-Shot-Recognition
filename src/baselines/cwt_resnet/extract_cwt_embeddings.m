function embeddings = extract_cwt_embeddings(featureExtractor,imageRoot, ...
    features,rows,hp,device)
% Extract fixed ResNet50 average-pool embeddings for test rows.

[files,~]=cwt_files_from_rows(imageRoot,features,rows);
assert(all(isfile(files)),'One or more requested CWT images are missing.');
batchSize=64;
embeddings=[];
for first=1:batchSize:numel(files)
    last=min(first+batchSize-1,numel(files));
    subset=files(first:last);
    images=zeros(hp.inputSize(1),hp.inputSize(2),3,numel(subset),'single');
    for i=1:numel(subset)
        value=imread(subset(i));
        if size(value,3)==1,value=repmat(value,1,1,3);end
        value=imresize(value,hp.inputSize);
        images(:,:,:,i)=single(value);
    end
    if string(device)=="gpu",images=gpuArray(images);end
    output=predict(featureExtractor,dlarray(images,'SSCB'));
    output=gather(extractdata(output));
    output=reshape(output,[],numel(subset))';
    if isempty(embeddings)
        embeddings=zeros(numel(files),size(output,2),'single');
    end
    embeddings(first:last,:)=single(output);
end
end
