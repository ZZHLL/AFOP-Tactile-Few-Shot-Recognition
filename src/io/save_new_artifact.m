function save_new_artifact(path, payload)
% Save an artifact to a new file.

assert(~isfile(path), 'Refusing to overwrite existing artifact: %s', path);
assert(isstruct(payload) && isscalar(payload), 'payload must be a scalar struct.');
parent = fileparts(path);
if ~exist(parent, 'dir')
    mkdir(parent);
end
save(path, '-struct', 'payload', '-v7.3');
end
