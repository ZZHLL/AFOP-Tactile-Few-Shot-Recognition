clear; clc;
% Evaluate four raw-signal baselines on a shared closed-set manifest.

root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'src'));
setup_afop_path();
mode = getenv_default('AFOP_MODE', 'pilot');
setenv('AFOP_MODE', mode);

families = ["feat", "mamlpp", "tactile_transformer", "channel_gat"];
for family = families
    fprintf('\n[Backbone evaluation] %s\n', family);
    switch family
        case "feat"
            run(fullfile(root, 'experiments', 'closed_set', 'evaluate_feat.m'));
        case "mamlpp"
            run(fullfile(root, 'experiments', 'closed_set', 'evaluate_mamlpp.m'));
        otherwise
            setenv('AFOP_MODEL_FAMILY', char(family));
            run(fullfile(root, 'experiments', 'closed_set', 'evaluate_support_only.m'));
    end
    setenv('AFOP_MODE', mode);
end

function value = getenv_default(name, fallback)
value = getenv(name);
if isempty(value), value = fallback; end
end
