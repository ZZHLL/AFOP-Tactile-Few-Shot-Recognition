clear; clc;
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(genpath(fullfile(root,'src'))); setup_afop_path();
cfg = benchmark_config(getenv_default('AFOP_MODE','full'));
data = load_project_data(cfg,true);
imageRoot = getenv_default('AFOP_CWT_IMAGE_ROOT',cfg.paperCWT.imageRoot);
existingCount = numel(dir(fullfile(imageRoot,'S*','*.jpg')));
if existingCount == 2160
    fprintf('[CWT] Existing 2160 images retained: %s\n',imageRoot); return;
end
assert(existingCount == 0, ...
    'CWT image root is incomplete (%d/2160). Use a new empty output path.',existingCount);
parameters = struct('waveletType',"amor",'voicesPerOctave',20);
create_cwt_images(data.raw,imageRoot,cfg.paperCWT.inputSize,parameters);
assert(numel(dir(fullfile(imageRoot,'S*','*.jpg'))) == 2160, ...
    'CWT image generation did not produce 2160 files.');

function value=getenv_default(name,fallback)
value=getenv(name);if isempty(value),value=fallback;end
end
