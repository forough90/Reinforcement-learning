function [v, b] = policy_td3_stub(state)
% Simple placeholder for TD3 actor: gentle mean reversion of (v,b).
% Signature you'll keep later: [v,b] = policy_td3_actor(state)
target_v = 6;  target_b = 0.50;
v = max(0, 0.95*state.v1 + 0.05*target_v);
b = min(max(0, 0.95*(state.v1*0+state.v2*0+0.0) + 0.05*target_b),1); %#ok<*MINV, *MAXV>
end
