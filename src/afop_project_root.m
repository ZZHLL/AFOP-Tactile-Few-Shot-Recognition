function root = afop_project_root()
% Return the repository root.

root = fileparts(fileparts(mfilename('fullpath')));
end
