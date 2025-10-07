function RB = rb_step3(capacity, sdim, adim)
% Simple circular replay buffer used for Step 3 sanity checks.
% API:
%   RB = rb_step3(capacity, sdim, adim)
%   RB.push(s, a, r, s2, d)
%   [S, A, R, S2, D] = RB.sample(B)
%   n = RB.size()
%   RB.dump(filepath)   % saves S,A,R,S2,D to .mat

RB.S  = zeros(sdim, capacity, 'single');
RB.A  = zeros(adim, capacity, 'single');
RB.R  = zeros(1,    capacity, 'single');
RB.S2 = zeros(sdim, capacity, 'single');
RB.D  = zeros(1,    capacity, 'single');
RB.i = 0; RB.n = 0; RB.cap = capacity;

    function push(s,a,r,s2,d)
        RB.i = mod(RB.i, RB.cap) + 1;
        RB.n = min(RB.n + 1, RB.cap);
        RB.S(:,RB.i)  = single(s(:));
        RB.A(:,RB.i)  = single(a(:));
        RB.R(:,RB.i)  = single(r);
        RB.S2(:,RB.i) = single(s2(:));
        RB.D(:,RB.i)  = single(d);
    end

    function [S,A,R,S2,D] = sample(B)
        if RB.n == 0, error('RB empty'); end
        idx = randi(RB.n, [B,1]);
        S  = RB.S(:, idx);
        A  = RB.A(:, idx);
        R  = RB.R(:, idx);
        S2 = RB.S2(:, idx);
        D  = RB.D(:, idx);
    end

    function n = size(), n = RB.n; end

    function dump(filepath)
        S  = RB.S(:,1:RB.n)';   %#ok<NASGU>
        A  = RB.A(:,1:RB.n)';   %#ok<NASGU>
        R  = RB.R(:,1:RB.n)';   %#ok<NASGU>
        S2 = RB.S2(:,1:RB.n)';  %#ok<NASGU>
        D  = RB.D(:,1:RB.n)';   %#ok<NASGU>
        if ~exist(fileparts(filepath),'dir'), mkdir(fileparts(filepath)); end
        save(filepath,'S','A','R','S2','D','-v7.3');
    end

RB.push  = @push;
RB.sample= @sample;
RB.size  = @size;
RB.dump  = @dump;
end
