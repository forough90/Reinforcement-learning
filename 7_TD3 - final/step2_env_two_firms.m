function Env = step2_env_two_firms(P)
% STEP 2 — RL-ready two-firm labor-market environment
% Firm 1: RL-controlled  (action a = [dv; db] per-step deltas)
% Firm 2: Trend-Follower (bounded-rational rule using fill-rate & wage gaps)

if nargin==0, P = struct; end

% -------------------- Defaults / Params --------------------
if ~isfield(P,'T'),        P.T = 480;      end
if ~isfield(P,'nf'),       P.nf = 2;       end
if ~isfield(P,'seed'),     P.seed = 42;    end

% matching / econ
if ~isfield(P,'A'),        P.A = 0.471;    end
if ~isfield(P,'alpha'),    P.alpha = 0.60; end
if ~isfield(P,'lambda'),   P.lambda = 0.10;end
if ~isfield(P,'z'),        P.z = 500;      end
if ~isfield(P,'p1'),       P.p1 = 1000;    end
if ~isfield(P,'p2'),       P.p2 = 1000;    end
if ~isfield(P,'c'),        P.c = 100;      end

% initial conditions
if ~isfield(P,'U0'),  P.U0 = 300; end
if ~isfield(P,'e10'), P.e10 = 5;  end
if ~isfield(P,'e20'), P.e20 = 5;  end
if ~isfield(P,'v10'), P.v10 = 6;  end
if ~isfield(P,'v20'), P.v20 = 6;  end
if ~isfield(P,'b10'), P.b10 = 0.50; end
if ~isfield(P,'b20'), P.b20 = 0.50; end

% trend follower tuning
if ~isfield(P,'mu_b'),     P.mu_b = 0.20;  end
if ~isfield(P,'mu_v'),     P.mu_v = 0.05;  end
if ~isfield(P,'db_max_tf'),P.db_max_tf = 0.02; end
if ~isfield(P,'dv_max_tf'),P.dv_max_tf = 0.10; end
if ~isfield(P,'tau'),      P.tau = 100;    end

% RL action bounds
if ~isfield(P,'dv_max'),   P.dv_max = 0.10; end
if ~isfield(P,'db_max'),   P.db_max = 0.02; end

% reward scaling
if ~isfield(P,'reward_scale'), P.reward_scale = 1e-3; end

Env.P = P;

% -------------------- Internal State -----------------------
rng(P.seed);
S = []; t = 1; doneFlag = false; logs = struct();
Wavg_s = [];  % smoothed avg wage

% -------------------- API --------------------
Env.reset     = @reset_fn;
Env.step      = @step_fn;
Env.get_state = @() get_state(t);
Env.Sdim      = 10;
Env.Adim      = 2;
Env.Bounds    = struct('dv',[-P.dv_max P.dv_max], 'db',[-P.db_max P.db_max]);

% ===========================================================
% RESET
% ===========================================================
    function s0 = reset_fn()
        rng(P.seed);
        t = 1; doneFlag = false;
        T = P.T;

        % preallocate arrays
        S.U    = zeros(T,1);   S.U(1)    = P.U0;
        S.Wavg = zeros(T,1);   S.Theta   = zeros(T,1);

        S.v1 = zeros(T,1); S.v1(1) = P.v10;
        S.v2 = zeros(T,1); S.v2(1) = P.v20;

        S.b1 = zeros(T,1); S.b1(1) = P.b10;
        S.b2 = zeros(T,1); S.b2(1) = P.b20;

        S.w1 = zeros(T,1); S.w1(1) = wage(S.b1(1),P.z,P.p1);
        S.w2 = zeros(T,1); S.w2(1) = wage(S.b2(1),P.z,P.p2);

        S.e1 = zeros(T,1); S.e1(1) = P.e10;
        S.e2 = zeros(T,1); S.e2(1) = P.e20;

        S.m1 = zeros(T,1); S.m2 = zeros(T,1);
        S.r1 = zeros(T,1); S.r2 = zeros(T,1);
        S.fr1= zeros(T,1); S.fr2= zeros(T,1);

        S.Wavg(1) = (S.w1(1)*S.e1(1) + S.w2(1)*S.e2(1)) / max(S.e1(1)+S.e2(1),eps);
        S.Theta(1)= (S.v1(1)+S.v2(1)) / max(S.U(1),eps);

        Wavg_s = S.Wavg(1);
        s0 = get_state(1);
    end

% ===========================================================
% STEP
% ===========================================================
    function [s1, r, done, info] = step_fn(a)
        if doneFlag
            s1 = get_state(t); r=0; done=true; info=build_info(true); return;
        end
        if nargin<1 || isempty(a), a = [0;0]; end

        % prev stocks
        U0=S.U(t); e10=S.e1(t); e20=S.e2(t);
        v10=S.v1(t); v20=S.v2(t); w10=S.w1(t); w20=S.w2(t);

        sep1 = P.lambda*e10; sep2=P.lambda*e20;
        Ueff = U0+sep1+sep2;

        % RL firm 1
        dv=max(-P.dv_max,min(P.dv_max,a(1)));
        db=max(-P.db_max,min(P.db_max,a(2)));
        v1=max(0,v10+dv);
        b1=clamp01(S.b1(t)+db);
        w1=wage(b1,P.z,P.p1);

        % Trend Follower
        Vtot_prev=max(v10+v20,0);
        MktFill_prev=(S.m1(max(t-1,1))+S.m2(max(t-1,1)))/(Vtot_prev+eps);
        fr2_prev=S.m2(max(t-1,1))/(v20+eps);
        dE2=fr2_prev-MktFill_prev;
        Wavg_s=0.95*Wavg_s+0.05*S.Wavg(t);
        dW2=w20-Wavg_s;
        db2=clip(P.mu_b*dE2,P.db_max_tf)+0.1*clip(dW2/P.tau,P.db_max_tf/2);
        dv2=clip(P.mu_v*dE2,P.dv_max_tf);
        b2=clamp01(S.b2(t)+db2);
        v2=max(0,v20+dv2);
        w2=wage(b2,P.z,P.p2);

        % Matching
        VTOT=max(v1+v2,0);
        Mraw=P.A*(Ueff^P.alpha)*(max(VTOT,0)^(1-P.alpha));
        Mtot=min([Mraw,Ueff,VTOT]);
        [m1,m2]=allocate_matches_nf(Mtot,Ueff,[v1,v2],[w1,w2],P.nf,P.tau);
        m1=min(m1,v1); m2=min(m2,v2);
        s=m1+m2; if s>Mtot+1e-9, sc=Mtot/s; m1=sc*m1; m2=sc*m2; end

        % stocks
        e1=max(0,e10-sep1+m1);
        e2=max(0,e20-sep2+m2);
        U=max(0,U0+sep1+sep2-m1-m2);

        raw_r1=-(P.c*v1)+(P.p1-w1)*e1;
        r1=P.reward_scale*raw_r1;

        Wavg=(w1*e1+w2*e2)/max(e1+e2,eps);
        fr1=m1/(v1+eps); fr2=m2/(v2+eps); Theta=(v1+v2)/max(U,eps);

        % commit
        S.v1(t+1)=v1; S.v2(t+1)=v2;
        S.b1(t+1)=b1; S.b2(t+1)=b2;
        S.w1(t+1)=w1; S.w2(t+1)=w2;
        S.e1(t+1)=e1; S.e2(t+1)=e2;
        S.U(t+1)=U; S.m1(t+1)=m1; S.m2(t+1)=m2;
        S.Wavg(t+1)=Wavg; S.fr1(t+1)=fr1; S.fr2(t+1)=fr2;
        S.Theta(t+1)=Theta; S.r1(t+1)=r1;

        t=t+1; doneFlag=(t>=P.T);

        s1=get_state(t);
        r=S.r1(t);
        done=doneFlag;
        info=build_info(doneFlag);
    end

% ===========================================================
% HELPERS
% ===========================================================
    function s=get_state(k)
        s=double([S.U(k),S.e1(k),S.e2(k),S.v1(k),S.v2(k), ...
                  S.m1(k),S.m2(k),S.Wavg(k),S.w1(k),S.w2(k)]);
    end

    function info=build_info(is_terminal)
        info=struct(); info.r1=S.r1(t); info.Wavg=S.Wavg(t);
        info.fr1=S.fr1(t); info.fr2=S.fr2(t); info.s_t=get_state(t);
        if is_terminal
            flds={'v1','v2','w1','w2','b1','b2','e1','e2','m1','m2','r1','U','Wavg','Theta','fr1','fr2'};
            L=struct(); for i=1:numel(flds),f=flds{i};L.(f)=double(S.(f)(1:t));end
            info.logs=L;
        end
    end
end

% utilities
function w=wage(b,z,p), w=b*z+(1-b)*p; end
function x=clamp01(x), x=max(0,min(1,x)); end
function y=clip(x,m), y=max(-m,min(m,x)); end

function [m1,m2]=allocate_matches_nf(Mtot,U,V,W,nf,tau)
    if sum(V)<=0||U<=0||Mtot<=0, m1=0; m2=0; return; end
    V=max(V,0); W=max(W,0);
    if nf<=1
        share=V/sum(V);
    else
        sW=exp((W-max(W))/tau);
        att=V.*(sW.^nf);
        if sum(att)==0, share=V/sum(V); else, share=att/sum(att); end
    end
    m1=Mtot*share(1); m2=Mtot*share(2);
end
