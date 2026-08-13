function root = setup_afop_path()
% Add the open-source package to the MATLAB path.

root = afop_project_root();
addpath(genpath(fullfile(root, 'src')), '-begin');
addpath(fullfile(root, 'configs'), '-begin');
addpath(root, '-begin');
end
