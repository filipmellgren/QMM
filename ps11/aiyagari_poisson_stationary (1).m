%Transition after unexpected decrease in aggregate productivity ("MIT shock")

clear all; clc; close all;

tic;

ga = 2;
rho = 0.05;
rho = 0.005;
d = 0.05;
d = 0.025;
al = 1/3;
Aprod = 0.1;
z1 = 1;
z2 = 2*z1;
z = [z1,z2];
la1 = 1/3;
la2 = 1/3;
la = [la1,la2];
z_ave = (z1*la2 + z2*la1)/(la1 + la2);

T = 200;
% N = 400;
N = 800;
dt = T/N;
time = (0:N-1)*dt;
max_price_it = 300;
convergence_criterion = 10^(-5);

% Asset grid
I= 1000;
amin = -0.8;
% amin = 0.0;
amax = 20;
a = linspace(amin,amax,I)';
da = (amax-amin)/(I-1);

aa = [a,a];
zz = ones(I,1)*z;

% Some convergence parameters
maxit= 100;
crit = 10^(-6);
Delta = 1000;
Ir = 40;
crit_S = 10^(-5);

% Declarations
dVf = zeros(I,2);
dVb = zeros(I,2);
c = zeros(I,2);

Aswitch = [-speye(I)*la(1),speye(I)*la(1);speye(I)*la(2),-speye(I)*la(2)];


% Generating initial values
r = 0.04;
w = 0.05;
v0(:,1) = (w*z(1) + r.*a).^(1-ga)/(1-ga)/rho;
v0(:,2) = (w*z(2) + r.*a).^(1-ga)/(1-ga)/rho;

% starting point, upper and lower bound for interest rate bisection
r0 = 0.03;
% rmin = 0.01;
rmin = 0.0001;
rmax = 0.99*rho;


%%%%%%%%%%%%%%%%
% STEADY STATE %
%%%%%%%%%%%%%%%%


for ir=1:Ir

    r_r(ir)=r;
    rmin_r(ir)=rmin;
    rmax_r(ir)=rmax;

    KD(ir) = (al*Aprod/(r + d))^(1/(1-al))*z_ave;
    w = (1-al)*Aprod*KD(ir).^al*z_ave^(-al);

    if ir>1
    v0 = V_r(:,:,ir-1);
    end

    v = v0;

    for n=1:maxit
        V = v;
        % forward difference
        dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVf(I,:) = (w*z + r.*amax).^(-ga); %will never be used, but impose state constraint a<=amax just in case
        % backward difference
        dVb(2:I,:) = (V(2:I,:)-V(1:I-1,:))/da;
        dVb(1,:) = (w*z + r.*amin).^(-ga); %state constraint boundary condition

        %consumption and savings with forward difference
        cf = dVf.^(-1/ga);
        ssf = w*zz + r.*aa - cf;
        %consumption and savings with backward difference
        cb = dVb.^(-1/ga);
        ssb = w*zz + r.*aa - cb;
        
        % When backward and forward difference is not applicable, we use steady state
        % Below is consumption and derivative of value function at steady state
        c0 = w*zz + r.*aa;
        dV0 = c0.^(-ga);

        % dV_upwind makes a choice of forward or backward differences based on
        % the sign of the drift 
        % Indicators
        If = ssf > 0; %positive drift --> forward difference
        Ib = ssb < 0; %negative drift --> backward difference
        I0 = (1-If-Ib); %at steady state

        % Upwind and consumption
        dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0; %important to include third term
        c = dV_Upwind.^(-1/ga);
        u = c.^(1-ga)/(1-ga);

        % CONSTRUCT A MATRIX
        X = -min(ssb,0)/da;
        Y = -max(ssf,0)/da + min(ssb,0)/da;
        Z = max(ssf,0)/da;

        A1=spdiags(Y(:,1),0,I,I)+spdiags(X(2:I,1),-1,I,I)+spdiags([0;Z(1:I-1,1)],1,I,I);
        A2=spdiags(Y(:,2),0,I,I)+spdiags(X(2:I,2),-1,I,I)+spdiags([0;Z(1:I-1,2)],1,I,I);
        A = [A1,sparse(I,I);sparse(I,I),A2] + Aswitch;
        
        % CONSTRUCT B MATRIX, where Delta is for XXX
        B = (1/Delta + rho)*speye(2*I) - A;

        % Stacking and finding solution for valuessolution
        u_stacked = [u(:,1);u(:,2)];
        V_stacked = [V(:,1);V(:,2)];

        b = u_stacked + V_stacked/Delta;
        V_stacked = B\b; %SOLVE SYSTEM OF EQUATIONS

        % Updating value
        V = [V_stacked(1:I),V_stacked(I+1:2*I)];

        % Compute critical value
        Vchange = V - v;
        v = V;

        dist(n) = max(max(abs(Vchange)));
        
        % Checking convergence
        if dist(n)<crit
            disp('Value Function Converged, Iteration = ')
            disp(n)
            break
        end
    end
    toc;

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Computing distribution %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    AT = A';
    b = zeros(2*I,1);

    %need to fix one value, otherwise matrix is singular
    i_fix = 1;
    b(i_fix)=.1;
    row = [zeros(1,i_fix-1),1,zeros(1,2*I-i_fix)];
    AT(i_fix,:) = row;

    %Solve linear system
    gg = AT\b;
    g_sum = gg'*ones(2*I,1)*da;
    gg = gg./g_sum;

    g = [gg(1:I),gg(I+1:2*I)];

    check1 = g(:,1)'*ones(I,1)*da;
    check2 = g(:,2)'*ones(I,1)*da;

    g_r(:,:,ir) = g;
    adot(:,:,ir) = w*zz + r.*aa - c;
    V_r(:,:,ir) = V;

    % Aggregating assets
    KS(ir) = g(:,1)'*a*da + g(:,2)'*a*da;
    
    % Computing excess supply of capital
    S(ir) = KS(ir) - KD(ir);

    % Bisection - updating interest rate
    if S(ir)>crit_S
        disp('Excess Supply')
        rmax = r;
        r = 0.5*(r+rmin);
    elseif S(ir)<-crit_S
        disp('Excess Demand')
        rmin = r;
        r = 0.5*(r+rmax);
    elseif abs(S(ir))<crit_S
        display('Equilibrium Found, Interest rate =')
        disp(r)
        break
    end
end

