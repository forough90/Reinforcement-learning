% smoketest
% OUT = step4_td3_learn('NEP',6,'Tmax',300,'Warmup',1500,'ReplayCap',5e4,...
%                       'Batch',256,'LR',3e-4,'Tau',0.005,'Gamma',0.99,...
%                       'NoiseAct',0.10,'Seed',42,'nf',2);

function OUT = step4_td3_learn(varargin)
% STEP 4 — TD3 for Firm 1 on the two-firm DMP-style env.
% Actions are [dv, db] in [-1,1]^2; env converts with step sizes.

% -------- Args --------
ip = inputParser;
addParameter(ip,'NEP',5);
addParameter(ip,'Tmax',300);
addParameter(ip,'Warmup',2000);
addParameter(ip,'ReplayCap',5e4);
addParameter(ip,'Batch',256);
addParameter(ip,'LR',3e-4);
addParameter(ip,'Tau',0.005);
addParameter(ip,'Gamma',0.99);
addParameter(ip,'NoiseAct',0.10);
addParameter(ip,'NoisePolicy',0.20);
addParameter(ip,'NoiseClip',0.50);
addParameter(ip,'Delay',2);
addParameter(ip,'Seed',42);
addParameter(ip,'nf',2);
addParameter(ip,'SaveAs','results/td3_agent.mat');
addParameter(ip,'DoPlots',true);
parse(ip,varargin{:});
P = ip.Results;

rng(P.Seed);

% -------- Env --------
E = struct('T',P.Tmax,'nf',P.nf,'seed',P.Seed);
Env = step2_env_two_firms(E);
sdim = 10; adim = 2;

% -------- Replay Buffer --------
RB = rb_init(P.ReplayCap, sdim, adim);

% -------- Networks --------
[actor, q1, q2] = build_models(sdim, adim);
tactor = clone_net(actor);
tq1    = clone_net(q1);
tq2    = clone_net(q2);

% -------- Optims --------
optA  = adamopt(P.LR);
optQ1 = adamopt(P.LR);
optQ2 = adamopt(P.LR);

% -------- Training --------
total_steps = 0;
ret_hist = zeros(P.NEP,1);

for ep = 1:P.NEP
    s = Env.reset();
    ret = 0;

    for t = 1:P.Tmax
        total_steps = total_steps + 1;

        % --- action selection ---
        if total_steps <= P.Warmup
            a = max(-1, min(1, randn(2,1)*0.5));
        else
            a = gather(exploit_action(actor, s));
            a = a + P.NoiseAct*randn(2,1);
            a = max(-1, min(1, a));
        end

        % --- env step ---
        [sp, r, done, info] = Env.step(a); %#ok<NASGU>
        RB = rb_add(RB, s(:), a(:), r(1), sp(:), done);
        ret = ret + r(1);
        s = sp;

        % --- learn ---
        if RB.size >= max(P.Batch, P.Warmup)
            B = rb_sample(RB, P.Batch);

            dlS  = dlarray(single(B.S),  'CB');   % (sdim,N)
            dlA  = dlarray(single(B.A),  'CB');
            dlSp = dlarray(single(B.Sp), 'CB');
            dlR  = dlarray(single(B.R),  'CB');
            dlD  = dlarray(single(B.D),  'CB');

            % Targets with policy smoothing (vectorized, label-safe)
            a_targ = fwd(tactor, dlSp);                                 % [-1,1]
            noise  = dlarray(P.NoisePolicy * randn(size(a_targ), ...
                     'like', extractdata(a_targ)), 'CB');
            noise  = max(-P.NoiseClip, min(P.NoiseClip, noise));
            a_targ = max(-1, min(1, a_targ + noise));

            q1_tp1 = fwd(tq1, dlSp, a_targ);
            q2_tp1 = fwd(tq2, dlSp, a_targ);
            q_tp1  = min(q1_tp1, q2_tp1);

            y = dlR + P.Gamma .* (1-single(dlD)) .* q_tp1;

            % Critics
            [~, gradQ1] = dlfeval(@criticLoss, q1, dlS, dlA, y);
            [~, gradQ2] = dlfeval(@criticLoss, q2, dlS, dlA, y);
            q1 = adamstep(q1, gradQ1, optQ1);
            q2 = adamstep(q2, gradQ2, optQ2);

            % Delayed actor + target updates
            if mod(total_steps, P.Delay) == 0
                [~, gradA] = dlfeval(@actorLoss, actor, q1, dlS);
                actor = adamstep(actor, gradA, optA);

                tactor = softUpdate(tactor, actor, P.Tau);
                tq1    = softUpdate(tq1,    q1,    P.Tau);
                tq2    = softUpdate(tq2,    q2,    P.Tau);
            end
        end

        if done, break; end
    end

    ret_hist(ep) = ret;
    if P.DoPlots
        fprintf('EP %d/%d  return=%.3g  RB=%d\n', ep, P.NEP, ret, RB.size);
    end
end

% -------- Save & output --------
if ~isempty(P.SaveAs)
    [d,~,~] = fileparts(P.SaveAs); if ~isempty(d) && ~exist(d,'dir'), mkdir(d); end
    save(P.SaveAs,'actor','q1','q2','tactor','tq1','tq2','RB','P','ret_hist','-v7.3');
end

if P.DoPlots
    figure('Color','w','Position',[70 70 1100 430]);
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
    nexttile; plot(ret_hist,'-o'); grid on; title('Episode return'); xlabel('ep');
    Bshow = rb_sample(RB, min(3000,RB.size));
    nexttile; scatter(Bshow.A(1,:), Bshow.A(2,:), 6, Bshow.R(:),'filled'); grid on;
    xlabel('dv'); ylabel('db'); title('Actions (colored by r)');
    nexttile; plot(movmean(Bshow.R, max(10,round(numel(Bshow.R)/50)))); grid on;
    title('RB rewards (moving mean)'); xlabel('sample');
end

OUT = struct('actor',actor,'critic1',q1,'critic2',q2, ...
             'tactor',tactor,'tcritic1',tq1,'tcritic2',tq2, ...
             'RB',RB,'ret_hist',ret_hist,'params',P);

% ========= local fns (main) =========
function a = exploit_action(actor, s)
    dlS = dlarray(single(s(:)), 'CB');   % (sdim,1)
    a = extractdata(fwd(actor, dlS));
end

end % ===== end main =====


% ======================================================================
% =                           MODELS                                   =
% ======================================================================
function [actor, q1, q2] = build_models(sdim, adim)
layersA = [
    featureInputLayer(sdim, 'Name','s')
    fullyConnectedLayer(128, 'Name','fc1')
    leakyReluLayer(0.2,'Name','lrelu1')
    fullyConnectedLayer(128, 'Name','fc2')
    leakyReluLayer(0.2,'Name','lrelu2')
    fullyConnectedLayer(adim, 'Name','fc_out')
    tanhLayer('Name','tanh')];
lgraphA = layerGraph(layersA);
actor   = dlnetwork(lgraphA);

sa = sdim + adim;
layersQ = [
    featureInputLayer(sa,'Name','sa')
    fullyConnectedLayer(256,'Name','qfc1')
    leakyReluLayer(0.2,'Name','qlrelu1')
    fullyConnectedLayer(256,'Name','qfc2')
    leakyReluLayer(0.2,'Name','qlrelu2')
    fullyConnectedLayer(1,'Name','qout')];
lgraphQ = layerGraph(layersQ);
q1 = dlnetwork(lgraphQ);
q2 = dlnetwork(lgraphQ);
end

% Unified forward (avoids recursion and release issues)
function y = fwd(net, s, a)
% Actor: y = fwd(actor, s)
% Critic: y = fwd(q, s, a)  -> concatenates along 'C'
if nargin==3
    x = [s; a];
    y = forward(net, x);
else
    y = forward(net, s);
end
end

% --------- Losses ---------
function [L, gradQ] = criticLoss(q, dlS, dlA, ytarget)
qpred = fwd(q, dlS, dlA);
L = mean((qpred - ytarget).^2,'all');       % inline MSE
gradQ = dlgradient(L, q.Learnables);
end

function [L, gradA] = actorLoss(actor, q1, dlS)
a = fwd(actor, dlS);
Q = fwd(q1, dlS, a);
L = -mean(Q);
gradA = dlgradient(L, actor.Learnables);
end

% --------- Target/Clone helpers ---------
function tgt = softUpdate(tgt, src, tau)
Lt = tgt.Learnables; Ls = src.Learnables;
for i=1:size(Lt,1)
    Lt.Value{i} = (1-tau)*Lt.Value{i} + tau*Ls.Value{i};
end
tgt = dlnetwork(layerGraph(tgt));
end

function net2 = clone_net(net1)
net2 = dlnetwork(layerGraph(net1));   % new object, same topology
Lt = net2.Learnables; Ls = net1.Learnables;
for i = 1:size(Lt,1)
    Lt.Value{i} = Ls.Value{i};
end
net2 = dlnetwork(layerGraph(net2));
end

% --------- Adam optimizer (robust numeric) ---------
function opt = adamopt(lr)
opt.lr   = lr;
opt.beta1= 0.9; opt.beta2=0.999; opt.eps=1e-8;
opt.t = containers.Map('KeyType','char','ValueType','int32');
opt.m = containers.Map('KeyType','char','ValueType','any');
opt.v = containers.Map('KeyType','char','ValueType','any');
end

function net = adamstep(net, grads, opt)
% Robust Adam step:
% - numeric tensors
% - empty/mismatched grads handled
% - reshape grads when numel matches param
% - reinit optimizer state if param size changes
L = net.Learnables;
for i = 1:size(L,1)
    name = char(L.Parameter(i));

    % Parameter tensor (numeric)
    W = L.Value{i};
    if isa(W,'dlarray'), W = extractdata(W); end

    % Gradient tensor (numeric, aligned to W)
    g = grads.Value{i};
    if isempty(g)
        g = zeros(size(W), 'like', W);
    else
        if isa(g,'dlarray'), g = extractdata(g); end
        if ~isequal(size(g), size(W))
            if numel(g)==numel(W)
                g = reshape(g, size(W));
            else
                g = zeros(size(W), 'like', W); % skip if incompatible
            end
        end
    end

    % Optional stability: gradient clip
    g = max(min(g, single(1.0)), single(-1.0));

    % Initialize or resize optimizer state
    newShape = (~isKey(opt.t,name));
    if ~newShape
        if ~isequal(size(opt.m(name)), size(W)) || ~isequal(size(opt.v(name)), size(W))
            newShape = true;
        end
    end
    if newShape
        opt.t(name) = int32(0);
        opt.m(name) = zeros(size(W), 'like', W);
        opt.v(name) = zeros(size(W), 'like', W);
    end

    % Load state
    t = double(opt.t(name)) + 1; opt.t(name) = int32(t);
    m = opt.m(name); v = opt.v(name);

    % Adam update
    beta1=opt.beta1; beta2=opt.beta2; eps=opt.eps; lr=opt.lr;
    m = beta1.*m + (1-beta1).*g;
    v = beta2.*v + (1-beta2).*(g.^2);
    mhat = m ./ (1 - beta1^t);
    vhat = v ./ (1 - beta2^t);
    W = W - lr .* mhat ./ (sqrt(vhat) + eps);

    % Save
    L.Value{i} = W;
    opt.m(name) = m; opt.v(name) = v;
end
net = dlnetwork(layerGraph(net));  % refresh internal state
end

% ======================================================================
% =                        ENV + RB (embedded)                         =
% ======================================================================
function Env = step2_env_two_firms(P)
% TD3 uses dv,db in [-1,1]; Env converts them with step sizes eta_v, eta_b.
% Trend follower is sign-correct: underfill -> wage up, vacancies down.
d = struct('A',0.471,'alpha',0.60,'lambda',0.10,'z',500,'p1',1000,'p2',1000,...
           'c',100,'U0',300,'e10',5,'e20',5,'T',480,'nf',2,'seed',42,'tau',75,...
           'eta_v',0.5,'eta_b',0.05, ...
           'trend',struct('mu_b',0.50,'mu_v',0.10,'db_max',0.05,'dv_max',0.25));
P = filldefaults(P,d);
S = struct(); rng(P.seed);

Env.reset = @reset; Env.step = @step;

    function w = wage(b,z,p), w = b*z + (1-b)*p; end
    function x = clamp01(x), x = max(0,min(1,x)); end
    function y = clip(x,m),  y = max(-m, min(m, x)); end
    function m = matches(u,v)
        if u<=0 || v<=0, m=0; else
            mraw = P.A*(u^P.alpha)*(max(v,0)^(1-P.alpha));
            m = min([mraw,u,v]);
        end
    end
    function [m1,m2] = alloc(Mtot,U,V,W,nf,tau)
        if sum(V)<=0 || U<=0 || Mtot<=0, m1=0; m2=0; return; end
        V = max(V,0); W = max(W,0);
        if nf<=1, share = V/sum(V);
        else
            sW = exp((W - max(W))/tau); att = V.*(sW.^nf);
            if sum(att)<=0, share = V/sum(V); else, share = att/sum(att); end
        end
        m1=Mtot*share(1); m2=Mtot*share(2);
        m1=min(m1,V(1)); m2=min(m2,V(2)); s=m1+m2;
        if s>0 && abs(s-Mtot)>1e-10, sc=Mtot/s; m1=sc*m1; m2=sc*m2; end
    end
    function s = obs()
        w1 = wage(S.b1,P.z,P.p1); w2 = wage(S.b2,P.z,P.p2);
        Wavg = (w1*S.e1 + w2*S.e2)/max(S.e1+S.e2,eps);
        s = [S.U, S.e1, S.e2, S.v1, S.v2, S.m1, S.m2, Wavg, w1, w2];
    end

    function s0 = reset()
        S.t=1; S.U=P.U0; S.e1=P.e10; S.e2=P.e20;
        S.v1=6; S.b1=0.5; S.v2=6; S.b2=0.5; S.m1=0; S.m2=0;
        s0=obs();
    end

    function [s1, r, done, info] = step(a)
        a = max(-1,min(1,a(:)));
        S.v1 = max(0, S.v1 + P.eta_v * a(1));
        S.b1 = clamp01(S.b1 + P.eta_b * a(2));
        w1 = wage(S.b1,P.z,P.p1);

        % Trend follower (sign-correct)
        w2  = wage(S.b2,P.z,P.p2);
        Vtp = max(S.v1+S.v2, eps);
        mkt = (S.m1+S.m2)/Vtp;
        fr2 = S.m2/(S.v2+eps);
        dE  = fr2 - mkt;                    % underfill < 0
        Wav = (w1*S.e1 + w2*S.e2)/max(S.e1+S.e2,eps);
        dW  = w2 - Wav;

        db = clip(+P.trend.mu_b * dE, P.trend.db_max);
        dv = clip(+P.trend.mu_v * dE, P.trend.dv_max);
        db = db + 0.2*clip(+dW/P.tau, P.trend.db_max/2);

        S.b2 = clamp01(S.b2 + db); S.v2 = max(0, S.v2 + dv);
        w2   = wage(S.b2,P.z,P.p2);

        VTOT = S.v1 + S.v2; Mtot = matches(S.U, VTOT);
        [m1,m2] = alloc(Mtot, S.U, [S.v1,S.v2], [w1,w2], P.nf, P.tau);

        e1p = max(0, S.e1 + m1 - P.lambda*S.e1);
        e2p = max(0, S.e2 + m2 - P.lambda*S.e2);
        Up  = max(0, S.U  + P.lambda*S.e1 + P.lambda*S.e2 - m1 - m2);

        S.m1=m1; S.m2=m2; S.e1=e1p; S.e2=e2p; S.U=Up; S.t=S.t+1;

        r1 = -(P.c*S.v1) + (P.p1 - w1)*S.e1;
        r2 = -(P.c*S.v2) + (P.p2 - w2)*S.e2;

        s1   = obs(); r=[r1 r2];
        done = S.t > P.T;
        info = struct('r1',r1,'r2',r2);
    end
end

% ---------------- Replay Buffer ----------------
function RB = rb_init(capacity, sdim, adim)
RB.cap=capacity; RB.sdim=sdim; RB.adim=adim;
RB.ptr=1; RB.size=0;
RB.S=zeros(sdim,capacity,'single'); RB.A=zeros(adim,capacity,'single');
RB.R=zeros(1,capacity,'single');    RB.Sp=zeros(sdim,capacity,'single');
RB.D=false(1,capacity);
end
function RB = rb_add(RB, s, a, r, sp, done)
i = RB.ptr;
RB.S(:,i)=single(s); RB.A(:,i)=single(a); RB.R(1,i)=single(r);
RB.Sp(:,i)=single(sp); RB.D(1,i)=logical(done);
RB.ptr = i+1; if RB.ptr>RB.cap, RB.ptr=1; end
RB.size = min(RB.size+1, RB.cap);
end
function batch = rb_sample(RB, n)
n = min(RB.size, n);
idx = randi(RB.size,[1 n]);
batch.S = RB.S(:,idx); batch.A = RB.A(:,idx); batch.R = RB.R(:,idx);
batch.Sp = RB.Sp(:,idx); batch.D = RB.D(:,idx);
end

% ---------------- Misc ----------------
function s = filldefaults(s, d)
f = fieldnames(d);
for k=1:numel(f)
    if ~isfield(s,f{k}) || isempty(s.(f{k})), s.(f{k}) = d.(f{k}); end
end
end
