% Run Env with random actions to validate shapes & plots for nf=1,2,20
clear; clc; close all;

T  = 480;
nfs = [1 2 20];

outs = cell(1,numel(nfs));
for k=1:numel(nfs)
    P = struct('T',T,'nf',nfs(k));
    Env = step2_env_two_firms(P);
    s = Env.reset();

    % random policy (bounded); TD3 will replace this next step
    for t=1:T-1
        a = [0.10*randn; 0.02*randn];   % ~N(0, sigma) within bounds
        [~, ~, ~, info] = Env.step(a);
    end
    outs{k} = info.logs;  % collect episode logs
end

% ---- plots (blue = RL, green = Trend) ----
col1 = [0 0.2 0.9]; col2 = [0 0.6 0];

figure('Color','w','Position',[40 40 1200 580]);
tl = tiledlayout(4,3,'TileSpacing','compact','Padding','compact');

titles = {'n_f = 1','n_f = 2','n_f = 20'};
for j=1:numel(nfs)
    L = outs{j};

    nexttile; hold on;
        plot(L.v1,'-','Color',col1,'LineWidth',1.5);
        plot(L.v2,'-','Color',col2,'LineWidth',1.5);
        grid on; title(['Vacancies — ',titles{j}]); xlabel('Time');
        if j==1, legend('RL (Firm 1)','Trend (Firm 2)','Location','best'); end

    nexttile; hold on;
        plot(L.w1,'-','Color',col1,'LineWidth',1.5);
        plot(L.w2,'-','Color',col2,'LineWidth',1.5);
        grid on; title('Wages'); xlabel('Time');

    nexttile; hold on;
        plot(L.e1,'-','Color',col1,'LineWidth',1.5);
        plot(L.e2,'-','Color',col2,'LineWidth',1.5);
        grid on; title('Employed Workers'); xlabel('Time');

    nexttile; hold on;
        plot(L.r1,'-','Color',col1,'LineWidth',1.5);
        grid on; title('Rewards (Firm 1)'); xlabel('Time');
end
title(tl,'Step 2 — RL-ready Env (Blue = RL, Green = Trend)');
