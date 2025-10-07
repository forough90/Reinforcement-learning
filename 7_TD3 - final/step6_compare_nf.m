function OUT = step6_compare_nf(varargin)
% STEP 6 — Multi-seed evaluation + paper-style figure

ip = inputParser;
addParameter(ip,'AgentFile','');               % OR Actor+Bounds
addParameter(ip,'Actor',[]);
addParameter(ip,'Bounds',[]);                  % [dvb dbb] -> scale tanh outputs
addParameter(ip,'Tmax',480);
addParameter(ip,'nf_list',[1 2 20]);
addParameter(ip,'Seeds',42);
addParameter(ip,'Summary','rl');               % 'rl' or 'total'
addParameter(ip,'WavgWindow',50);
addParameter(ip,'SaveFig','');
parse(ip,varargin{:});
R = ip.Results;

% ----- resolve actor + bounds -----
[actor, dvb, dbb] = resolve_actor_and_bounds(R.AgentFile, R.Actor, R.Bounds);

nf_list = R.nf_list(:)';  nfL = numel(nf_list);
seeds   = R.Seeds(:)';

% ----- rollouts (nf × seed) -----
ALL = cell(nfL, numel(seeds));
for i = 1:nfL
    for j = 1:numel(seeds)
        % FIX: env expects T (not Tmax)
        P = struct('T',R.Tmax,'nf',nf_list(i),'seed',seeds(j));
        ALL{i,j} = rollout_one(actor, P, dvb, dbb);
    end
end

% ----- aggregate (mean across seeds) -----
Agg = cell(nfL,1);
for i = 1:nfL
    Agg{i} = mean_logs(ALL(i,:));
end

% ----- summary values for left panel -----
final_employed = zeros(nfL,1);
final_wavg     = zeros(nfL,1);
for i = 1:nfL
    L = Agg{i}; T = numel(L.e1);
    if strcmpi(R.Summary,'rl')
        final_employed(i) = L.e1(T);                 % Firm 1 only
    else
        final_employed(i) = L.e1(T) + L.e2(T);       % Total
    end
    t0 = max(1, T - R.WavgWindow + 1);
    final_wavg(i) = mean(L.Wavg(t0:T));              % smoothed wage
end

% ----- figure (rows = variables, cols = nf) -----
f = figure('Color','w','Position',[30 30 1300 720]);

cols = 1 + nfL;                              % 1 summary column + nf columns
tl = tiledlayout(4, cols, 'Padding','compact', 'TileSpacing','compact');
title(tl,'RL Agent (blue) vs Trend Follower (red) — Averaged over seeds');

% Left summary column (spans all rows)
axL = nexttile([4 1]); hold(axL,'on'); grid(axL,'on');
x = 1:nfL; set(axL,'XTick',x,'XTickLabel',compose('n_f = %d', nf_list));
yyaxis(axL,'left');
plot(axL, x, final_employed, 'd','MarkerSize',10, 'MarkerFaceColor',[0 0.6 0],'MarkerEdgeColor','k');
ylabel(axL, sprintf('Employed %s', tern(strcmpi(R.Summary,'rl'),'(Firm 1)','(Total)')));
yyaxis(axL,'right');
plot(axL, x, final_wavg, 'p','MarkerSize',10, 'MarkerFaceColor',[0.55 0 0.8],'MarkerEdgeColor','k');
ylabel(axL,'Average Wage');
title(axL,'Summary (final values)');
legend(axL,{'Employed','Average Wage'},'Location','best');

% ---- rows = variables; cols = nf ----
col_offset = 1;                              % col #1 used by summary
rowlabels  = {'Vacancies','Wages','Employed','Rewards'};

for j = 1:nfL
    L = Agg{j}; t = 1:numel(L.v1);

    % Row 1: Vacancies (tile = (row-1)*cols + (col_offset+j))
    ax = nexttile( (1-1)*cols + (col_offset + j) );
    plot(ax,t,L.v1,'b','LineWidth',1.2); hold(ax,'on');
    plot(ax,t,L.v2,'r','LineWidth',1.2); grid(ax,'on');
    if j==1, ylabel(ax,rowlabels{1}); end
    title(ax, sprintf('n_f = %d', nf_list(j)));
    if j==2, legend(ax,{'RL Agent','Trend Follower'},'Location','best'); end

    % Row 2: Wages
    ax = nexttile( (2-1)*cols + (col_offset + j) );
    plot(ax,t,L.w1,'b','LineWidth',1.2); hold(ax,'on');
    plot(ax,t,L.w2,'r','LineWidth',1.2); grid(ax,'on');
    if j==1, ylabel(ax,rowlabels{2}); end

    % Row 3: Employed Workers
    ax = nexttile( (3-1)*cols + (col_offset + j) );
    plot(ax,t,L.e1,'b','LineWidth',1.2); hold(ax,'on');
    plot(ax,t,L.e2,'r','LineWidth',1.2); grid(ax,'on');
    if j==1, ylabel(ax,rowlabels{3}); end

    % Row 4: Rewards
    ax = nexttile( (4-1)*cols + (col_offset + j) );
    plot(ax,t,L.r1,'b','LineWidth',1.2); hold(ax,'on');
    plot(ax,t,L.r2,'r','LineWidth',1.2); grid(ax,'on');
    if j==1, ylabel(ax,rowlabels{4}); end
    xlabel(ax,'Time');
end

if ~isempty(R.SaveFig)
    [pth,~,~]=fileparts(R.SaveFig);
    if ~isempty(pth) && ~isfolder(pth), mkdir(pth); end
    exportgraphics(f,R.SaveFig,'Resolution',200);
end

% ----- out -----
OUT.ALL = ALL;
OUT.Agg = Agg;
OUT.final_employed = final_employed;
OUT.final_wavg     = final_wavg;
OUT.fig = f;
OUT.params = R;
end

% ================= helpers =================
function [actor, dvb, dbb] = resolve_actor_and_bounds(agentFile, actorIn, boundsIn)
if ~isempty(actorIn) && ~isempty(boundsIn)
    actor = actorIn; dvb = boundsIn(1); dbb = boundsIn(2); return;
end
if isempty(agentFile)
    error('Provide AgentFile OR Actor+Bounds.');
end
S = load(agentFile); actor = [];
if isfield(S,'actor'), actor = S.actor; end
if isempty(actor) && isfield(S,'OUT') && isfield(S.OUT,'actor'), actor = S.OUT.actor; end
if isempty(actor)
    error('Actor not found in %s. Available vars: %s', agentFile, strjoin(fieldnames(S),', '));
end
if isempty(boundsIn)
    error('Bounds [dvb dbb] missing. Call with Bounds matching Step-4 env (e.g., [0.5 0.05]).');
end
dvb = boundsIn(1); dbb = boundsIn(2);
end

function OUT = rollout_one(actor, P, dvb, dbb)
Env = step2_env_two_firms(P);   % P must contain T, nf, seed
s = Env.reset();
% FIX: use P.T (not P.Tmax)
if isfield(P,'T'), T = P.T; else, T = 480; end

log = struct('v1',zeros(T,1,'single'),'v2',zeros(T,1,'single'), ...
             'w1',zeros(T,1,'single'),'w2',zeros(T,1,'single'), ...
             'e1',zeros(T,1,'single'),'e2',zeros(T,1,'single'), ...
             'r1',zeros(T,1,'single'),'r2',zeros(T,1,'single'), ...
             'U',zeros(T,1,'single'),'Wavg',zeros(T,1,'single'));
for t=1:T
    u = predict_actor(actor, s);        % numeric [-1,1]^2
    a = [u(1)*dvb; u(2)*dbb];           % scale to env increments
    [s, r, done, info] = Env.step(a);

    % state: [U e1 e2 v1 v2 m1 m2 Wavg w1 w2]
    U_    = s(1); e1_ = s(2); e2_ = s(3);
    v1_   = s(4); v2_ = s(5);
    Wavg_ = s(8); w1_ = s(9); w2_ = s(10);
    if numel(r)==2, r1_ = r(1); r2_ = r(2); else, r1_ = r; r2_ = NaN; end

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
    if done, T=t; break; end
end
fn = fieldnames(log); for k=1:numel(fn), log.(fn{k})=log.(fn{k})(1:T); end
OUT.log = log;
end

function Lm = mean_logs(rowcells)
flds = fieldnames(rowcells{1}.log);
Tmin = inf;
for j=1:numel(rowcells), Tmin = min(Tmin, numel(rowcells{j}.log.(flds{1}))); end
Lm = struct();
for f=1:numel(flds)
    M = zeros(Tmin, numel(rowcells), 'single');
    for j=1:numel(rowcells)
        v = rowcells{j}.log.(flds{f}); M(:,j) = v(1:Tmin);
    end
    Lm.(flds{f}) = mean(M,2,'omitnan');
end
end

function u = predict_actor(actor, s_row)
x = dlarray(single(s_row(:)),'CB');      % [D x 1]
y = forward(actor, x);                   % [2 x 1], linear
u = tanh(y);
u = double(gather(extractdata(u)));      % numeric [-1,1]^2
end

function out = tern(cond,a,b)
if cond, out=a; else, out=b; end
end

function v = getfield_default(S, name, default)
if isstruct(S) && isfield(S,name), v = S.(name); else, v = default; end
end
