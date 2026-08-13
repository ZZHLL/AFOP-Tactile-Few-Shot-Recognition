function run = create_run_directory(cfg, modelName, stage)
% Create a timestamped run directory.

if nargin < 3 || isempty(stage)
    stage = "train";
end
stamp = string(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
run.id = matlab.lang.makeValidName(modelName + "_" + stage + "_" + stamp);
run.dir = fullfile(cfg.paths.runs, char(run.id));
assert(startsWith(string(run.dir), string(cfg.root), 'IgnoreCase', true), ...
    'Refusing to write outside the isolated workspace.');
assert(~exist(run.dir, 'dir'), 'Run directory already exists: %s', run.dir);
mkdir(run.dir);
run.checkpoints = fullfile(run.dir, 'checkpoints');
run.logs = fullfile(run.dir, 'logs');
run.results = fullfile(run.dir, 'results');
mkdir(run.checkpoints);
mkdir(run.logs);
mkdir(run.results);
end
