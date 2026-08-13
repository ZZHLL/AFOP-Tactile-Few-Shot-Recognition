function path = find_latest_protocol_artifact(artifactsRoot)
% Return the newest closed-set protocol artifact.

matches = dir(fullfile(artifactsRoot, 'closedset_protocol_*.mat'));
assert(~isempty(matches), 'No closed-set protocol artifact found.');
[~, newest] = max([matches.datenum]);
path = fullfile(matches(newest).folder, matches(newest).name);
end
