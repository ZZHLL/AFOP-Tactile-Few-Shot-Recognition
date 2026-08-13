function params = gather_mamlpp_parameters(params)
% Gather explicit MAML parameters for portable checkpoint storage.

params = move_mamlpp_parameters(params, "cpu");
end
