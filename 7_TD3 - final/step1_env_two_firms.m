function OUT = step1_env_two_firms(varargin)
% STEP 1 — Minimal two-firm labor market simulator (adjusted)

ip = inputParser;
addParameter(ip,'T',480);
addParameter(ip,'nf',2);
addParameter(ip,'seed',42);

% econ params
addParameter(ip,'A',0.471);
addParameter(ip,'alpha',0.60);
addParameter(ip,'lambda',0.10);
addParameter(ip,'z',500);
addParameter(ip,'p1',1000);
addParameter(ip,'p2',1000);
addParameter(ip,'c',100);

% initial conditions / policies
addParameter(ip,'U0',300);
addParameter(ip,'e10',5);
addParameter(ip,'e20',5);
addParameter(ip,'v10',6);
addParameter(ip,'v20',6);
addParameter(ip,'b10',0.50);
addParameter(ip,'b20',0.50);

% trend-follower tuning (ADJUSTED)
addParameter(ip,'mu_b',0.20);
addParameter(ip,'mu_v',0.05);
addParameter(ip,'db_max',0.02);
addParameter(ip,'dv_max',0.10);
addParameter(ip,'tau',100);   % reduced from 120

parse(ip,varargin{:});
rng(ip.Results.seed);

T   = ip.Results.T;     nf    = ip.Results.nf;
A   = ip.Results.A;     alpha = ip.Results.alpha;  lambda = ip.Results.lambda;
z   = ip.Results.z;     p1    = ip.Results.p1;     p2     = ip.Results.p2;
c   = ip.Results.c;     tau   = ip.Results.tau;

U   = zeros(T,1);   U(1) = ip.Results.U0;
Wavg= zeros(T,1);   Theta = zeros(T,1);

v1  = zeros(T,1); v2  = zeros(T,1);
b1  = zeros(T,1); b2  = zeros(T,1);
w1  = zeros(T,1); w2  = zeros(T,1);
e1  = zeros(T,1); e2  = zeros(T,1);
m1  = zeros(T,1); m2  = zeros(T,1);
r1  = zeros(T,1); r2  = zeros(T,1);
fr1 = zeros(T,1); fr2 = zeros(T,1);

v1(1)=ip.Results.v10;    v2(1)=ip.Results.v20;
b1(1)=ip.Results.b10;    b2(1)=ip.Results.b20;
w1(1)= wage(b1(1),z,p1); w2(1)= wage(b2(1),z,p2);
e1(1)= ip.Results.e10;   e2(1)= ip.Results.e20;
Wavg(1) = (w1(1)*e1(1)+w2(1)*e2(1)) / max(e1(1)+e2(1),eps);
Theta(1)= (v1(1)+v2(1)) / max(U(1),eps);

% --- ADJUSTED: smoothed average wage for TF reaction ---
Wavg_s = Wavg(1);

violations = 0; warned = false;

for t = 2:T
    sep1 = lambda*e1(t-1);
    sep2 = lambda*e2(t-1);
    Ueff = U(t-1) + sep1 + sep2;

    % Firm 1 (fixed slot)
    v1(t) = v1(t-1);
    b1(t) = clamp01(b1(t-1));
    w1(t) = wage(b1(t),z,p1);

    % Firm 2 (Trend Follower)
    Vtot_prev   = max(v1(t-1)+v2(t-1), 0);
    MktFill_prev= (m1(t-1)+m2(t-1)) / (Vtot_prev + eps);
    fr2_prev    = m2(t-1) / (v2(t-1) + eps);
    dE2 = fr2_prev - MktFill_prev;
    dW2 = w2(t-1) - Wavg_s;          % ADJUSTED: use smoothed Wavg

    db = clip(+ip.Results.mu_b * dE2, ip.Results.db_max);
    dv = clip(+ip.Results.mu_v * dE2, ip.Results.dv_max);
    % ADJUSTED: wage-gap kick uses smoothed average wage
    db = db + 0.2*clip(+dW2/tau, ip.Results.db_max/2);

    b2(t) = clamp01(b2(t-1) + db);
    v2(t) = max(0,   v2(t-1) + dv);
    w2(t) = wage(b2(t),z,p2);

    % Matching
    VTOT = max(v1(t)+v2(t), 0);
    Mraw = A * (Ueff^alpha) * (max(VTOT,0)^(1-alpha));
    Mtot = min([Mraw, Ueff, VTOT]);

    [m1(t), m2(t)] = allocate_matches_nf(Mtot, Ueff, [v1(t),v2(t)], [w1(t),w2(t)], nf, tau);
    m1(t) = min(m1(t), v1(t));  m2(t) = min(m2(t), v2(t));
    s = m1(t)+m2(t);
    if s > Mtot + 1e-10
        sc = Mtot / s; m1(t) = sc*m1(t); m2(t) = sc*m2(t);
    end

    % Update stocks
    e1(t) = max(0, e1(t-1) - sep1 + m1(t));
    e2(t) = max(0, e2(t-1) - sep2 + m2(t));
    U(t)  = max(0, U(t-1) + sep1 + sep2 - m1(t) - m2(t));

    % Rewards
    r1(t) = -(c*v1(t)) + (p1 - w1(t))*e1(t);
    r2(t) = -(c*v2(t)) + (p2 - w2(t))*e2(t);

    % Aggregates
    Wavg(t) = (w1(t)*e1(t) + w2(t)*e2(t)) / max(e1(t)+e2(t), eps);
    % ADJUSTED: update smoothed average wage
    Wavg_s  = 0.95*Wavg_s + 0.05*Wavg(t);

    fr1(t)  = m1(t) / (v1(t) + eps);
    fr2(t)  = m2(t) / (v2(t) + eps);
    Theta(t)= (v1(t)+v2(t)) / max(U(t),eps);

    if any([m1(t) > v1(t)+1e-8, m2(t) > v2(t)+1e-8, ...
            (m1(t)+m2(t)) > min(Ueff,VTOT)+1e-8])
        violations = violations + 1;
        if ~warned
            warning('Invariant(s) violated at t=%d. Simulation will continue.', t);
            warned = true;
        end
    end
end

OUT.t = (1:T)';
OUT.U = U;     OUT.Wavg = Wavg;  OUT.Theta = Theta;
OUT.logs = struct( ...
    'v1',v1,'v2',v2,'w1',w1,'w2',w2,'b1',b1,'b2',b2, ...
    'e1',e1,'e2',e2,'m1',m1,'m2',m2,'r1',r1,'r2',r2, ...
    'fr1',fr1,'fr2',fr2,'nf',nf, ...
    'A',A,'alpha',alpha,'lambda',lambda,'z',z,'p1',p1,'p2',p2,'c',c, ...
    'policy1',"fixed");
OUT.invariants.violations = violations;
end

% ---------- helpers ----------
function w = wage(b,z,p), w = b*z + (1-b)*p; end
function x = clamp01(x), x = max(0,min(1,x)); end
function y = clip(x, m), y = max(-m, min(m, x)); end

function [m1, m2] = allocate_matches_nf(Mtot, U, V, W, nf, tau)
    if sum(V)<=0 || U<=0 || Mtot<=0
        m1=0; m2=0; return;
    end
    V = max(V,0); W = max(W,0);
    if nf<=1
        share = V / sum(V);
    else
        sW = exp((W - max(W)) / tau);
        att = V .* (sW.^nf);
        if sum(att)==0, share = V/sum(V); else, share = att / sum(att); end
    end
    m1 = Mtot * share(1);
    m2 = Mtot * share(2);
end
