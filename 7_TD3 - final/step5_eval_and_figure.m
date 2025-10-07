function OUT = step5_eval_and_figure(varargin)
% STEP 5 — Deterministic evaluation & figure

ip = inputParser;
addParameter(ip,'AgentFile','');
addParameter(ip,'Actor',[]);
addParameter(ip,'Bounds',[]);           % [dvb dbb]
addParameter(ip,'Tmax',480);
addParameter(ip,'nf_list',[1 2 20]);
addParameter(ip,'Seed',42);
addParameter(ip,'SaveFig','');
parse(ip,varargin{:});
R = ip.Results; rng(R.Seed);

[actor, dvb, dbb] = resolve_actor_and_bounds(R.AgentFile, R.Actor, R.Bounds);

nf_list = R.nf_list(:)';
EVAL = cell(numel(nf_list),1);
for i = 1:numel(nf_list)
    % Step-2 env expects P.T, not P.Tmax
    P = struct('T',R.Tmax,'nf',nf_list(i),'seed',R.Seed);
    EVAL{i} = rollout_trajectories(actor, P, dvb, dbb);
end




% --- colors (RL blue, Trend green) ---
colRL   = [0 0.2 0.9];
colTrend= [0 0.6 0];

f = figure('Color','w','Position',[40 40 1200 720]);
tl = tiledlayout(4, numel(nf_list), 'Padding','compact','TileSpacing','compact');
title(tl,'RL Agent (blue) vs Trend Follower (green) — Evaluation');

rowtitles = {'Vacancies','Wages','Employed Workers','Rewards'};
for j = 1:numel(nf_list)
    L = EVAL{j}.log; T = numel(L.v1); t = 1:T;

    nexttile; hold on;
    plot(t,L.v1,'-','Color',colRL,'LineWidth',1.2);
    plot(t,L.v2,'-','Color',colTrend,'LineWidth',1.2);
    grid on; title(sprintf('%s — n_f = %d',rowtitles{1}, nf_list(j)));
    xlabel('Time'); ylabel('v');
    if j==1, legend({'RL Agent','Trend Follower'},'Location','best'); end

    nexttile; hold on;
    plot(t,L.w1,'-','Color',colRL,'LineWidth',1.2);
    plot(t,L.w2,'-','Color',colTrend,'LineWidth',1.2);
    grid on; title(rowtitles{2}); xlabel('Time'); ylabel('w');

    nexttile; hold on;
    plot(t,L.e1,'-','Color',colRL,'LineWidth',1.2);
    plot(t,L.e2,'-','Color',colTrend,'LineWidth',1.2);
    grid on; title(rowtitles{3}); xlabel('Time'); ylabel('e');

    nexttile; hold on;
    plot(t,L.r1,'-','Color',colRL,'LineWidth',1.2);
    plot(t,L.r2,'-','Color',colTrend,'LineWidth',1.2);
    grid on; title(rowtitles{4}); xlabel('Time'); ylabel('r');
end

if ~isempty(R.SaveFig)
    [pth,~,~]=fileparts(R.SaveFig);
    if ~isempty(pth) && ~isfolder(pth), mkdir(pth); end
    exportgraphics(f,R.SaveFig,'Resolution',200);
end

OUT.eval   = EVAL;
OUT.fig    = f;
OUT.actor  = actor;
OUT.bounds = [dvb dbb];
OUT.save   = R.SaveFig;
end


%% ---------- Helpers ----------
function [actor, dvb, dbb] = resolve_actor_and_bounds(agentFile, actorIn, boundsIn)
if ~isempty(actorIn) && ~isempty(boundsIn)
    actor = actorIn; dvb = boundsIn(1); dbb = boundsIn(2); return;
end
if isempty(agentFile)
    error('Provide AgentFile OR Actor+Bounds.');
end
S = load(agentFile); actor = []; dvb = []; dbb = [];

% Common save patterns
if isfield(S,'actor'), actor = S.actor; end
if isempty(actor) && isfield(S,'OUT') && isfield(S.OUT,'actor'), actor = S.OUT.actor; end

if isempty(actor)
    error('Actor not found in %s. Available vars: %s', agentFile, strjoin(fieldnames(S),', '));
end

if isempty(boundsIn)
    error(['Bounds [dvb dbb] not provided. Call with ''Bounds'',[eta_v eta_b] ', ...
           'matching Step-4/Step-2 env (e.g., [0.5 0.05]).']);
else
    dvb = boundsIn(1); dbb = boundsIn(2);
end
end


function OUT = rollout_trajectories(actor, P, dvb, dbb)
Env = step2_env_two_firms(P);
s = Env.reset();
T = P.T;                       % use P.T to match env

log = struct('v1',zeros(T,1,'single'),'v2',zeros(T,1,'single'), ...
             'w1',zeros(T,1,'single'),'w2',zeros(T,1,'single'), ...
             'e1',zeros(T,1,'single'),'e2',zeros(T,1,'single'), ...
             'r1',zeros(T,1,'single'),'r2',zeros(T,1,'single'), ...
             'U',zeros(T,1,'single'),'Wavg',zeros(T,1,'single'));

for t=1:T
    a = predict_actor(actor, s);          % [-1,1]^2
    a = [a(1)*dvb; a(2)*dbb];             % scale to env step sizes
    [s, r, done, info] = Env.step(a);

    % --- robust field extraction ---
    % state layout: [U e1 e2 v1 v2 m1 m2 Wavg w1 w2]
    U_    = s(1); e1_ = s(2); e2_ = s(3);
    v1_   = s(4); v2_ = s(5);
    Wavg_ = s(8); w1_ = s(9); w2_ = s(10);

    % rewards: r may be scalar (r1) or [r1 r2]
    if numel(r)==2, r1_ = r(1); r2_ = r(2);
    else,           r1_ = r;    r2_ = getfield_default(info,'r2',NaN);
    end

    % prefer info.* when present, otherwise use state-derived values
    log.v1(t)=single(getfield_default(info,'v1',v1_));
    log.v2(t)=single(getfield_default(info,'v2',v2_));
    log.w1(t)=single(getfield_default(info,'w1',w1_));
    log.w2(t)=single(getfield_default(info,'w2',w2_));
    log.e1(t)=single(getfield_default(info,'e1',e1_));
    log.e2(t)=single(getfield_default(info,'e2',e2_));
    log.r1(t)=single(getfield_default(info,'r1',r1_));
    log.r2(t)=single(getfield_default(info,'r2',r2_));
    log.U(t) =single(getfield_default(info,'U', U_));
    log.Wavg(t)=single(getfield_default(info,'Wavg',Wavg_));

    if done, T = t; break; end
end

% trim in case of early done
fn = fieldnames(log);
for k=1:numel(fn), log.(fn{k}) = log.(fn{k})(1:T); end

OUT.log = log;
OUT.return_rl = sum(double(log.r1));
OUT.return_tf = sum(double(log.r2(~isnan(log.r2))));
OUT.avg_wage  = mean(double(log.Wavg(max(1,T-50):T)));
end


function a = predict_actor(actor, s_row)
x = dlarray(single(s_row(:)),'CB');   % [D x 1]
y = forward(actor, x);                % [2 x 1]
a = tanh(y);                          % [-1,1]
a = reshape(a,2,1);
end

function v = getfield_default(S, name, default)
if isstruct(S) && isfield(S,name), v = S.(name); else, v = default; end
end
