function hp = feat_shot_hyperparameters(cfg, shot)
% Select shot-specific FEAT hyperparameters.

index = find(cfg.featOfficial.shots == shot, 1);
assert(~isempty(index), 'Unsupported FEAT shot count: %d.', shot);
hp.shot = shot;
hp.balance = cfg.featOfficial.balanceByShot(index);
hp.temperature = cfg.featOfficial.temperatureByShot(index);
hp.temperature2 = cfg.featOfficial.temperature2ByShot(index);
end
