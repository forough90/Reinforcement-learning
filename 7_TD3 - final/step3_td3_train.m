function OUT = step3_td3_train(varargin)
% STEP 3 — TD3 training for Firm 1 on step2_env_two_firms (manual-Adam version)
%
% Example (smoke):
% OUT = step3_td3_train('Episodes',12,'Tmax',480,'nf',2, ...
%     'Warmup',2000,'Batch',256,'LR',3e-4,'Seed',42, ...
%     'ReplayCap',1e5,'UpdatesPerStep',1,'doPlot',true);

%% ---------- Args ----------
ip = inputParser;
addParameter(ip,'Episodes',6);
addParameter(ip,'Tmax',480);
addParameter(ip,'nf',2);
addParameter(ip,'Warmup',2000);
addParameter(ip,'Batch',256);
addParameter(ip,'UpdatesPerStep',1);
addParameter(ip,'ReplayCap',1e5);
addParameter(ip,'LR',3e-4);
addParameter(ip,'Gamma',0.995);
addParameter(ip,'Tau',0.005);             % Polyak factor
addParameter(ip,'StdExpl',0.10);          % exploration noise (fraction of bounds)
addParameter(ip,'StdTarget',0.10);        % target smoothing noise (fraction of bounds)
addParameter(ip,'StdTargetClip',0.20);    % clip abs(noise) as fraction of bounds
addParameter(ip,'PolicyDelay',2);         % delayed actor updates
addParameter(ip,'Seed',42);
addParameter(ip,'doPlot',false);
parse(ip,varargin{:});
R = ip.Results; rng(R.Seed);

%% ---------- Environment ----------
% NOTE: if your Step-2 expects P.T rather than P.Tmax, we pass P.T.
P = struct('T',R.Tmax,'nf',R.nf,'seed',R.Seed);
Env = step2_env_two_firms(P);
s0  = Env.reset();                         % read state dimension / bounds

state_dim  = numel(s0);                    % [U e1 e2 v1 v2 m1 m2 Wavg w1 w2] => 10
act_dim    = 2;                            % [dv, db]

% Bounds: try to read from Env.P; if absent, fall back to sane defaults
if isfield(Env,'P') && isfield(Env.P,'dv_max') && isfield(Env.P,'db_max')
    dvb = Env.P.dv_max; dbb = Env.P.db_max;
else
    dvb = 0.10; dbb = 0.02;
end

%% ---------- Networks ----------
% Actor outputs in [-1,1]^2 (via tanh in predict_actor). Scale to bounds at the edge.
actor   = build_mlp(state_dim, [128 128], act_dim, 'linear');
critic1 = build_mlp(state_dim+act_dim, [128 128], 1, 'linear');
critic2 = build_mlp(state_dim+act_dim, [128 128], 1, 'linear');

tActor   = net_simple_clone(actor);
tCritic1 = net_simple_clone(critic1);
tCritic2 = net_simple_clone(critic2);

optA  = adam_init(actor,   R.LR);
optQ1 = adam_init(critic1, R.LR);
optQ2 = adam_init(critic2, R.LR);

%% ---------- Replay Buffer ----------
CAP = max(10000, min(R.ReplayCap, 2e6));
B = buffer_init(CAP, state_dim, act_dim);

%% ---------- Logs ----------
logs.ep_return = zeros(R.Episodes,1);
logs.avg_wage  = zeros(R.Episodes,1);
logs.ep_steps  = zeros(R.Episodes,1);

%% ---------- Training Loop ----------
gstep = 0;
for ep = 1:R.Episodes
    s = Env.reset();
    ret = 0; wsum = 0;

    for t = 1:R.Tmax
        gstep = gstep + 1;

        % ----- action -----
        if gstep <= R.Warmup
            a = [ (2*rand-1)*dvb; (2*rand-1)*dbb ];            % uniform in bounds
        else
            ah = predict_actor(actor, s);                       % 2x1 in [-1,1]
            a  = [ah(1)*dvb; ah(2)*dbb];                       % to bounds
            a  = a + (R.StdExpl*[dvb; dbb]).*randn(2,1,'single'); % exploration
            a(1)= clip(a(1), dvb); a(2)= clip(a(2), dbb);      % clip
        end

        % ----- env step -----
        [s2, r, done, info] = Env.step(a);
        ret  = ret + r;
        if isstruct(info) && isfield(info,'Wavg'), wsum = wsum + info.Wavg; end

        % store (actions scaled to [-1,1] for critics)
        a_scaled = [single(a(1)/max(dvb,eps)); single(a(2)/max(dbb,eps))]';
        buffer_push(B, s, a_scaled, r, s2, double(done));

        % ----- learn (only when buffer is ready) -----
        if gstep > R.Warmup && B.n >= R.Batch
            for k=1:R.UpdatesPerStep
                batch = buffer_sample(B, R.Batch);

                [critic1, optQ1, critic2, optQ2] = ...
                    td3_update_critics(critic1, critic2, tCritic1, tCritic2, tActor, ...
                        batch, R.Gamma, R.StdTarget, R.StdTargetClip, dvb, dbb, optQ1, optQ2);

                if mod(gstep, R.PolicyDelay)==0
                    [actor, optA] = td3_update_actor(actor, critic1, batch);
                    % Polyak (in-place)
                    tActor   = polyak_update(tActor, actor, R.Tau);
                    tCritic1 = polyak_update(tCritic1, critic1, R.Tau);
                    tCritic2 = polyak_update(tCritic2, critic2, R.Tau);
                end
            end
        end

        s = s2;
        if done, break; end
    end

    logs.ep_return(ep) = ret;
    logs.avg_wage(ep)  = wsum / max(t,1);
    logs.ep_steps(ep)  = t;

    if R.doPlot
        subplot(1,3,1); plot(logs.ep_return(1:ep),'-o'); grid on; title('Episode return'); xlabel ep;
        subplot(1,3,2); plot(logs.avg_wage(1:ep),'-o'); grid on; title('Avg wage'); xlabel ep;
        subplot(1,3,3); plot(logs.ep_steps(1:ep),'-o'); grid on; title('Steps'); xlabel ep; drawnow;
    end

    fprintf('EP %d/%d  return=%.2e  RB=%d\n', ep, R.Episodes, ret, B.n);
end

%% ---------- Output ----------
OUT.logs = logs;
OUT.actor   = actor;
OUT.critic1 = critic1; OUT.critic2 = critic2;
OUT.tActor  = tActor;  OUT.tCritic1 = tCritic1; OUT.tCritic2 = tCritic2;
OUT.bounds  = [dvb dbb];
OUT.params  = R;
end % ===== main =====


%% ===== Build tiny MLP as dlnetwork =====
function net = build_mlp(in_dim, h, out_dim, out_act) %#ok<INUSD>
% out_act kept 'linear'; we apply tanh in predict_actor for compatibility.
layers = [
    featureInputLayer(in_dim,'Name','in','Normalization','none')
    fullyConnectedLayer(h(1),'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(h(2),'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(out_dim,'Name','out')];
lgraph = layerGraph(layers);
net = dlnetwork(lgraph);
end

%% ===== Actor forward (2x1 in [-1,1]) =====
function a = predict_actor(actor, s_row)
% s_row: numeric 1xD -> dlarray [D x 1] with 'CB' labels
x = dlarray(single(s_row(:)),'CB');   % [D x 1]
y = forward(actor, x);                % [2 x 1], linear head
a = tanh(y);                          % ensure [-1,1]
a = reshape(a,2,1);
end

%% ===== Critic forward Q(s,a) =====
function q = predict_critic(critic, S_batch, A_batch)
% S_batch: BxD, A_batch: Bx2 (both numeric)
X = single([S_batch, A_batch]);   % B x (D+2)
X = dlarray(X','CB');             % [D+2 x B]
q = forward(critic, X);           % [1 x B]
q = q';                           % B x 1
end

%% ===== TD3 critic update (vectorized target smoothing) =====
function [critic1, optQ1, critic2, optQ2] = td3_update_critics(critic1, critic2, tCritic1, tCritic2, tActor, ...
        batch, gamma, stdT, stdClip, dvb, dbb, optQ1, optQ2)

S   = single(batch.S); 
A   = single(batch.A);            % [-1,1]
Rwd = single(batch.R);
S2  = single(batch.S2);
Dfl = single(batch.D);

% --- Vectorized target action with smoothing ---
% S2: B x D  -> X2: D x B
X2   = dlarray(S2','CB');                 % (D x B)
u2   = forward(tActor, X2);               % (2 x B), linear head
u2   = tanh(u2);                          % [-1,1]

% Scale to bounds
a2b  = [dvb; dbb] .* u2;                  % (2 x B) in physical bounds

% Add Gaussian smoothing noise, per-dim per-sample
noise = (stdT * [dvb; dbb]) .* randn(size(a2b), 'like', extractdata(a2b));
clipN =  (stdClip * [dvb; dbb]);
noise = max(min(noise, clipN), -clipN);

% Clip to bounds after noise
a2b = max(min(a2b + noise, [dvb; dbb]), -[dvb; dbb]);   % (2 x B)

% Back to [-1,1] range expected by critics
At = a2b ./ max([dvb; dbb], eps);         % (2 x B)
At = gather(extractdata(At))';            % -> (B x 2)

% TD target
Qt1  = predict_critic(tCritic1, S2, At);
Qt2  = predict_critic(tCritic2, S2, At);
Qtarg= Rwd + (1 - Dfl) .* gamma .* min(Qt1, Qt2);

% Critic-1 loss & grads
[gr1, ~] = dlfeval(@critic_grads, critic1, S, A, Qtarg);
% Critic-2 loss & grads
[gr2, ~] = dlfeval(@critic_grads, critic2, S, A, Qtarg);

% Adam updates (name-free via Learnables.Value)
[critic1, optQ1] = adam_step_net(critic1, gr1, optQ1);
[critic2, optQ2] = adam_step_net(critic2, gr2, optQ2);
end

function [grads, loss] = critic_grads(net, S, A, Qtarg)
Q = predict_critic(net, S, A);
loss = mean( (Q - dlarray(Qtarg)).^2 );
grads = dlgradient(loss, net.Learnables.Value);   % grads for learnables (cell array)
end

%% ===== TD3 actor update (delayed) =====
function [actor, optA] = td3_update_actor(actor, critic1, batch)
S = single(batch.S);  Bsz = size(S,1);
Ahat = zeros(Bsz,2,'single');
for i=1:Bsz
    ai = predict_actor(actor, S(i,:));            % 2x1 in [-1,1]
    Ahat(i,:) = single(gather(extractdata(ai)))';
end
[gr, ~] = dlfeval(@actor_grads, actor, critic1, S, Ahat);
[actor, optA] = adam_step_net(actor, gr, optA);
end

function [grads, loss] = actor_grads(actor, critic1, S, ~)
Bsz = size(S,1); A = zeros(Bsz,2,'single');
for i=1:Bsz
    ai = predict_actor(actor, S(i,:));
    A(i,:) = single(gather(extractdata(ai)))';
end
Q = predict_critic(critic1, S, A);
loss = -mean(Q);                                  % maximize Q
grads = dlgradient(loss, actor.Learnables.Value);
end

%% ===== Utilities: Polyak, Adam, clone, clip =====
function tnet = polyak_update(tnet, net, tau)
for i=1:height(net.Learnables)
    tnet.Learnables.Value{i} = (1-tau)*tnet.Learnables.Value{i} + tau*net.Learnables.Value{i};
end
end

function opt = adam_init(net, lr)
n = height(net.Learnables);
opt.m = cell(n,1); opt.v = cell(n,1);
for i=1:n
    sz = size(net.Learnables.Value{i});
    opt.m{i} = zeros(sz,'like',net.Learnables.Value{i});
    opt.v{i} = zeros(sz,'like',net.Learnables.Value{i});
end
opt.t = 0; opt.lr = lr; opt.beta1 = 0.9; opt.beta2 = 0.999; opt.eps = 1e-8;
end

function [net, opt] = adam_step_net(net, grads, opt)
opt.t = opt.t + 1; b1=opt.beta1; b2=opt.beta2; eps=opt.eps; lr=opt.lr;
for i=1:height(net.Learnables)
    g = grads{i};
    % gentle gradient clip for stability
    g = max(min(g, single(1.0)), single(-1.0));
    opt.m{i} = b1*opt.m{i} + (1-b1)*g;
    opt.v{i} = b2*opt.v{i} + (1-b2)*(g.^2);
    mhat = opt.m{i} / (1-b1^opt.t);
    vhat = opt.v{i} / (1-b2^opt.t);
    net.Learnables.Value{i} = net.Learnables.Value{i} - lr * mhat ./ (sqrt(vhat) + eps);
end
end

function net2 = net_simple_clone(net)
% Clone via layers + copy learnables (robust across releases).
lg = layerGraph(net.Layers);
net2 = dlnetwork(lg);
for i=1:height(net.Learnables)
    net2.Learnables.Value{i} = net.Learnables.Value{i};
end
end

function val = clip(x, b), val = max(-b, min(b, x)); end

%% ===== Replay buffer =====
function B = buffer_init(cap, sdim, adim)
B.cap = cap; B.sdim = sdim; B.adim = adim;
B.S  = zeros(cap, sdim, 'single');
B.A  = zeros(cap, adim, 'single');   % actions in [-1,1]
B.R  = zeros(cap, 1,     'single');
B.S2 = zeros(cap, sdim, 'single');
B.D  = zeros(cap, 1,     'single');
B.n  = 0; B.head = 1;
end

function buffer_push(B, s, a_scaled, r, s2, d)
i = B.head;
B.S(i,:)  = single(s(:)');
B.A(i,:)  = single(a_scaled);
B.R(i,1)  = single(r);
B.S2(i,:) = single(s2(:)');
B.D(i,1)  = single(d);
B.head = i + 1; if B.head > B.cap, B.head = 1; end
B.n = min(B.n+1, B.cap);
end

function batch = buffer_sample(B, Bsz)
% assumes B.n >= Bsz (enforced in training loop)
idx = randi(B.n, Bsz, 1);
batch.S  = B.S(idx,:);
batch.A  = B.A(idx,:);
batch.R  = B.R(idx,:);
batch.S2 = B.S2(idx,:);
batch.D  = B.D(idx,:);
end
