% smoke3_easy.m
% Ultra-light TD3 sanity check: tiny training, single rollout at n_f=2.

clear; clc; rng(42);

% --- 1) Tiny training run ---
OUT = step3_td3_train( ...
    'Episodes', 30, ...     % very small
    'Tmax',     300, ...
    'nf',       2, ...
    'Warmup',   800, ...
    'Batch',    128, ...
    'LR',       3e-4, ...
    'ReplayCap', 5e4, ...
    'UpdatesPerStep', 1, ...
    'PolicyDelay', 2, ...
    'Seed',     42, ...
    'doPlot',   true);      % will show 3 small curves (return, avg wage, steps)

actor = OUT.actor; dvb = OUT.bounds(1); dbb = OUT.bounds(2);

% --- 2) Single evaluation rollout at n_f=2 (paper-style 4 plots) ---
T = 300;
Env = step2_env_two_firms(struct('T',T,'nf',2,'seed',99));
s = Env.reset();
for t = 1:T-1
    a = policy_action(actor, s, dvb, dbb);   % deterministic actor
    [s, ~, done, info] = Env.step(a);
    if done, break; end
end
L = info.logs;

figure('Color','w','Position',[80 80 950 600]);
tiledlayout(2,2,'Padding','compact','TileSpacing','compact');

nexttile; plot(L.v1,'b','LineWidth',1.3); hold on; plot(L.v2,'r','LineWidth',1.3);
title('Vacancies (n_f=2)'); grid on;

nexttile; plot(L.w1,'b','LineWidth',1.3); hold on; plot(L.w2,'r','LineWidth',1.3);
title('Wages'); grid on;

nexttile; plot(L.e1,'b','LineWidth',1.3); hold on; plot(L.e2,'r','LineWidth',1.3);
title('Employed workers'); grid on;

nexttile; plot(L.r1,'b','LineWidth',1.3); hold on; if isfield(L,'r2'), plot(L.r2,'r','LineWidth',1.3); end
title('Rewards'); grid on; xlabel('Time');

legend({'RL (Firm 1)','Trend (Firm 2)'},'Location','southoutside','Orientation','horizontal');

% ---- helper ----
function a = policy_action(actor, s_row, dvb, dbb)
    x = dlarray(single(s_row(:)),'CB');
    u = tanh(forward(actor, x));      % [-1,1]^2
    u = gather(extractdata(u));
    a = [u(1)*dvb; u(2)*dbb];
end
