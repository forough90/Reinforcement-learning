% smoke1_step1.m
% Quick smoke test for step1_env_two_firms

clear; clc; rng(42);
nf_list = [1 2 20];
T = 500;

OUTS = cell(1,numel(nf_list));
for k = 1:numel(nf_list)
    OUTS{k} = step1_env_two_firms('T',T,'nf',nf_list(k));
    fprintf('nf=%d  invariant_violations=%d\n', nf_list(k), OUTS{k}.invariants.violations);
end

figure('Color','w','Position',[60 60 1200 700]);
tl = tiledlayout(4,3,'Padding','compact','TileSpacing','compact');

labels_col = {'n_f = 1','n_f = 2','n_f = 20'};

for c = 1:3
    L = OUTS{c}.logs;

    % Vacancies
    nexttile((1-1)*3+c);
    plot(L.v1,'-','LineWidth',1.3); hold on;
    plot(L.v2,'-','LineWidth',1.3);
    title(labels_col{c}); ylabel('Vacancies'); xlim([1 T]); grid on;

    % Wages
    nexttile((2-1)*3+c);
    plot(L.w1,'-','LineWidth',1.3); hold on;
    plot(L.w2,'-','LineWidth',1.3);
    ylabel('Wages'); xlim([1 T]); grid on;

    % Employed workers
    nexttile((3-1)*3+c);
    plot(L.e1,'-','LineWidth',1.3); hold on;
    plot(L.e2,'-','LineWidth',1.3);
    ylabel('Employed'); xlim([1 T]); grid on;

    % Rewards
    nexttile((4-1)*3+c);
    plot(L.r1,'-','LineWidth',1.3); hold on;
    plot(L.r2,'-','LineWidth',1.3);
    xlabel('Time'); ylabel('Rewards'); xlim([1 T]); grid on;
end

lg = legend({'RL slot','Trend follower'},'Orientation','horizontal');
lg.Layout.Tile = 'south';
title(tl,'Step 1 — Two-firm simulator (smoke test)');
