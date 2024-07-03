%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution of the homogeneous, Fisher-KPP equation, using a direct application of the Carleman method
% combined with Euler's method. The result is compared with solutions from inbuilt MATLAB solvers.
%
% Code written by Manish Kumar in 2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear global variables at the beginning
clear;


%% Plotting configuration
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
fontsize = 20;

%% Simulation parameters: SPACE TIME DISCRETIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nx = 8; % Spatial discretization for Euler's method for ODE
nt = 1000; % Temporal discretization for Euler's method for ODE

% for inbuilt PDE solver (Proxy for exact answer)
nx_pde = 100; % Spatial discretization for the 'pdepe' solver
nt_pde = nt*20; % Temporal discretization for the 'pdepe' solver

L0 = (1); % Domain length

% Spatial domain edges
x0 = 0;
x1 = L0;

% Temporal domain edges
T = (3);  % TOTAL Simulation time
t0 = (0); % START TIME
t1 = T;   % END TIME

%% REACTION DIFFUSION PARAMETERS

%Re0 = (20); % Desired Reynolds number
D = (0.2); % DIFFUSION
aa = (0.4); % LINEAR TERM in f= aa*u + bb*u^2
bb = (-1); % QUADRATIC TERM in f= aa*u + bb*u^2

%% CARLEMAN TRUNCATION ORDER
N_max = 3; % Maximum Carleman truncation level
ode_deg = 2; % Degree of the Carleman ODE, should not be changed

Ns = 1:N_max; % Truncation levels
nu = D; %VISCOCITY/DIFFUSION cefficient
beta = aa; %linear damping term


%% TAYLOR-method order and discretization of space and time
K_max = 3; % Taylor truncation order
Ks = 1:K_max;

dx = double((x1-x0)/(nx-1));
dt = double((t1-t0)/(nt-1));
xs = linspace(x0,x1,nx);
ts = linspace(t0,t1,nt);

%% RUNGE-KUTTA ode45 discretization: proxy for exact solver 
nt_ode = nt*10; % Make it more accurate than the Euler solution
dt_ode = double((t1-t0)/(nt_ode-1));
ts_ode = linspace(t0,t1,nt_ode)';

%% pdepe discretization interval sizes
dx_pde = (x1-x0)/(nx_pde-1); % Spatial discretization interval size for pdepe solver
dt_pde = (t1-t0)/(nt_pde-1);
xs_pde = linspace(x0,x1,nx_pde);
ts_pde = linspace(t0,t1,nt_pde);

%% Non-Homogenous terms
F0 = zeros(nt,nx);
%for it = 1:nt
 %   F0(it,:) = F0_fun(ts(it),xs); %time wise populate the F0 matrix
%end

%% Initial condition
U0 = double(0.1); % u(x,0)=0.1*(1-cos(2*pi*x))
f = 1; % Number of oscillations of initial distribution
u0 = @(x) U0*(sin((f*pi)*x/L0).^2);
u0s = u0(xs);
Tnl = L0/U0; % Nonlinear time
t_plot = 2*Tnl/5; % Time to plot solution

%% Linear term matrix (Shifted and Scaled LAPLACIAN)
F1 = zeros(nx, nx);
F1(1+1:nx+1:end) = D/dx^2;
F1(1+nx:nx+1:end) = D/dx^2;
F1(1:nx+1:end) = -(2)*D/dx^2;    %%%%%%%% Edited
F1 = F1 + aa*eye(nx); % Add linear damping if present

%% Quadratic term matrix
F2 = zeros(nx ,nx^2);
%F2((nx^2+nx+1):(nx^2+nx+1):end) = -1/(4*dx);  %%%%%%%%%%%%%%  edited
%F2(1+1:(nx^2+nx+1):end) = +1/(4*dx);           %%%%%%%%%%%%%  edited
F2(1:nx^2+nx+1:end)= bb;   %%%%%%%%%%%%%%%%%%  added
F2 = reshape(F2,nx, nx^2);

%% Enforce the Dirichlet boundaries within the domain.
% F0(1) = 0;
% F0(end) = 0;
F1(1, :) = 0; %first row
F1(end, :) = 0; %last row
F2(1, :) = 0; %first row
F2(end, :) = 0; %first row

%% ODE for ode45 solver
%F0_interp = @(t) interp1(ts,F0,t)'; %linear interpolation 
burgers_odefun = @(t,u) F1*u + F2*kron(u,u);

%% PDE, initial condition and boundary condition for pdepe solver
burger_pde = @(x,t,u,dudx) deal(1, nu*dudx, -beta*u);    %%%%%%%%
burger_ic = u0;
burger_bc = @(xl, ul, xr, ur, t) deal(ul, 0, ur, 0);

%% Check CFL condition
% Dissipative and advective CFL numbers for Euler's method and pdepe
C1_e = U0*dt/dx;
C2_e = 2*nu*dt/dx^2;
C1_ode = U0*dt_ode/dx;
C2_ode = 2*nu*dt_ode/dx^2;
C1_pde = U0*dt_pde/dx_pde;
C2_pde = 2*nu*dt_pde/dx_pde^2;

if C1_e > 1
    error(sprintf("C1_e = %.2f\n",C1_e));
end
if C2_e > 1
    error(sprintf("C2_e = %.2f\n",C2_e));
end
if C1_ode > 1
    error(sprintf("C1_ode = %.2f\n",C1_ode));
end
if C2_ode > 1
    error(sprintf("C2_ode = %.2f\n",C2_ode));
end
if C1_pde > 1
    error(sprintf("C1_pde = %.2f\n",C1_pde));
end
if C2_pde > 1
    error(sprintf("C2_pde = %.2f\n",C2_pde));
end


%% Calculate the Carleman convergence number

lambdas = eig(F1);
%fprintf('printing eigenvalues of F1\n');
%disp(lambdas);

% Since we included the Dirichlet boundaries explicitly, the matrix F1 has
% a 2D nullspace corresponding to eigenvectors with non-zero boundary
% values. We remove these two zero-eigenvalues on the next line of
% code. We could also have avoided this by not explicitly including the
% zeroed boundaries in the integration domain, which corresponds to reducing
% nx by 2. This would increase lambda_1 according to the formula
%
% lambda_j = -nu*4/dx^2*sin(j*pi/(2*(nx+1-2)))^2.
%
% Instead we choose to keep the boundaries explicitly in the domain and just
% discard the zero eigenvalues. This will result in a larger R, so
% the claims in the paper are conservative relative to this. Note that the
% spectral norm of F2 is only marginally changed by the zeroing of
% boundaries.

lambdas = lambdas(lambdas ~= 0);
lambda = max(lambdas);

f2 = norm(F2);
f1 = norm(F1);
f0 = 0;

for it = 1:nt
    f0 = max(norm(F0(it,:)),f0);
end

R = ( norm(u0s)*f2 + f0/norm(u0s) )/abs(lambda);

r1 = (abs(lambda)-sqrt(lambda^2-4*f2*f0))/(2*f2);
r2 = (abs(lambda)+sqrt(lambda^2-4*f2*f0))/(2*f2);

if dt > 1/(N_max*f1)
    error('Time step too large');
end

if f0 + f2 > abs(lambda)
    fprintf('Perturbation too large\n');
end

%% Prepare Carleman matrix
fprintf('Preparing Carleman matrix\n');

% Calculate matrix block sizes for carleman matrix
dNs = zeros(N_max, 1);

for N = Ns
    dNs(N) = (nx^(N+1)-nx)/(nx-1);
end

%% First prepare the Carleman system with just the source term at t=0
A = spalloc(dNs(end),dNs(end),dNs(end)*nx);

Fs = [F1 F2];

for i = Ns % carleman matrix block wise preparation
    for j = 1:min(ode_deg, N_max-i+1)
        if i == 1 && j == 0
            continue;
        end
        a0 = 1+(nx^i-nx)/(nx-1);
        a1 = a0 + nx^i-1;
        b0 = 1+(nx^(j+i-1)-nx)/(nx-1);
        b1 = b0 + nx^(j+i-1)-1;

        Aij = spalloc(nx^i,nx^(i+j-1),nx^(i+j-1+1));
        f0 = 1+(nx^j-nx)/(nx-1);
        f1 = f0+nx^j-1;
        Fj = Fs(:,f0:f1);

        for p = 1:i
            Ia = kronp(sparse(eye(nx)), p-1);
            Ib = kronp(sparse(eye(nx)), i-p);
            Aij = Aij + kron(kron(Ia, Fj), Ib);
        end
        A(a0:a1, b0:b1) = Aij;
    end
end


%eigensA = eig(A);
fprintf('matrix A size\n');
disp(size(A));

%%% Conjugate equation
% A = transpose(A);



%% Solve Carleman system using TAYLOR method
% preparing TAYLOR series terms

%WK = zeros(dNs(N), dNs(N)); % TYLOR product matrix

ys_c_N = zeros(K_max, nt, dNs(N_max));


for N = Ks
    A_N = A; % Truncated carleman matrix
    
    %%% preparing TAYLOR series terms
    WK = zeros(dNs(N_max), dNs(N_max)); % TYLOR product matrix initialize
    A_N_dt = double(dt*A_N); %time multiplied matrix
    
    %----------- timestamping for Taylor matrix buildup
    fprintf('Taylor matrix buildup start time\n');
    disp(datetime);
    
    for l = 0:N
        WK = WK + double(((A_N_dt)^l)*(1/factorial(l)));
    end
    
    fprintf('Taylor matrix buildup end time\n');
    disp(datetime);
    %----------- timestamping
    
    
    %% initial solution vector
    y0s = [];
    
    for i = 1:N_max
        y0s = [y0s, kronp(u0s, i)];
    end
    
    
    %---------timestamping for numerical integration
    fprintf('Solving Taylor Truncation for N=%d\n',N);

    ys = zeros(nt ,dNs(N_max));
    ys(1,:) = y0s;
    
    fprintf('Numerical integration loop start time\n');
    disp(datetime);
    for k = 1:(nt-1)
        ys(k+1,:) = (WK*ys(k,:)')';
    end
    
    fprintf('Numerical integration loop end time\n');
    disp(datetime);
    %---------timestamping for numerical integration
    
    fprintf('Grand loop Done----------------------\n');
    
    ys_c_N(N, : ,1:dNs(N_max)) = real(ys(:,:));
end

us_c_N = double(ys_c_N(:,:,1:nx)); % extracting solution from carleman Solution vector


%% Solve direct Euler
fprintf('Solving direct Euler\n');
us_e = zeros(nt,nx);
us_e(1,:) = u0s;

for k = 1:(nt-1)
    us_e(k+1,:) = us_e(k,:) + dt*burgers_odefun(ts(k),us_e(k,:)')';
end


%% Solve "exact" ODE
fprintf('Solving "exact" ODE\n');
opts = odeset('RelTol',1e-10,'AbsTol',1e-10);
[ts_ode, us_ode] = (ode45(burgers_odefun, ts_ode, u0s, opts));

% Interpolate so we can compare with other solutions
us_d = interp1(ts_ode,us_ode,ts);


%% Solve "exact" PDE
fprintf('Solving "exact" PDE\n');
us_pde = pdepe(0, burger_pde, burger_ic, burger_bc, xs_pde, ts_pde);

% Interpolate so we can compare with other solutions.
% First interpolate over space, and then over time.
us_pde_interp_temp = zeros(nt,nx_pde);
for i = 1:nx_pde
    us_pde_interp_temp(:,i) = interp1(ts_pde,us_pde(:,i),ts);
end
us_pde_interp = zeros(nt,nx);
for k = 1:nt
    us_pde_interp(k,:) = interp1(xs_pde,us_pde_interp_temp(k,:),xs);
end


%% Calculate errors
% We will now calculate the error between all three solutions. First find
% their differences, and then take their norms over space. The variables
% are named according to:
%
% dus_ = difference
% eps_ = l_2 error
% eps_rel_ = l_inf error
%
% _c_ = Carleman solution
% _e_ = Direct Euler solution
% _d_ = ode45 solution
% _pde_ = pdepe solution
%
% _N = per Carleman level N

dus_c_d_N = zeros(K_max,nt,nx);
dus_rel_c_d_N = zeros(K_max,nt,nx);
eps_c_d_N = zeros(K_max,nt);
eps_rel_c_d_N = zeros(K_max,nt);

dus_c_pde_N = zeros(K_max,nt,nx);
dus_rel_c_pde_N = zeros(K_max,nt,nx);
eps_c_pde_N = zeros(K_max,nt);
eps_rel_c_pde_N = zeros(K_max,nt);

dus_d_pde = zeros(nt,nx);
dus_rel_d_pde = zeros(nt,nx);
eps_d_pde = zeros(1,nt);
eps_rel_d_pde = zeros(1,nt);

dus_d_e = zeros(nt,nx);
dus_rel_d_e = zeros(nt,nx);
eps_d_e = zeros(1,nt);
eps_rel_d_e = zeros(1,nt);


% Type of norm
lp = 2; % Two norm

for N = 1:N_max
    dus_c_d_N(N,:,:) = reshape(us_c_N(N,:,:),nt,nx)-us_d(:,:);
    dus_rel_c_d = reshape(dus_c_d_N(N,:,:),nt,nx)./us_d(:,:);
    dus_rel_c_d(isnan(dus_rel_c_d)) = 0;
    dus_rel_c_d_N(N,:,:) = dus_rel_c_d;

    dus_c_pde_N(N,:,:) = reshape(us_c_N(N,:,:),nt,nx)-us_pde_interp(:,:);
    dus_rel_c_pde = reshape(dus_c_pde_N(N,:,:),nt,nx)./us_pde_interp(:,:);
    dus_rel_c_pde(isnan(dus_rel_c_pde)) = 0;
    dus_rel_c_pde(isinf(dus_rel_c_pde)) = 0;
    dus_rel_c_pde_N(N,:,:) = dus_rel_c_pde;

    dus_d_pde(:,:) = reshape(us_d(:,:),nt,nx)-us_pde_interp(:,:);
    dus_rel_d_pde = reshape(dus_d_pde(:,:),nt,nx)./us_pde_interp(:,:);
    dus_rel_d_pde(isnan(dus_rel_d_pde)) = 0;
    dus_rel_d_pde(isinf(dus_rel_d_pde)) = 0;

    dus_d_e(:,:) = reshape(us_d(:,:),nt,nx)-us_e(:,:);
    dus_rel_d_e = reshape(dus_d_e(:,:),nt,nx)./us_e(:,:);
    dus_rel_d_e(isnan(dus_rel_d_e)) = 0;
    dus_rel_d_e(isinf(dus_rel_d_e)) = 0;

    for k = 1:nt
        eps_c_d_N(N,k) = norm(reshape(dus_c_d_N(N,k,:),nx,1), lp); %%% NORM %%%
        eps_rel_c_d_N(N,k) = norm(reshape(dus_rel_c_d_N(N,k,:),nx,1), lp);

        eps_c_pde_N(N,k) = norm(reshape(dus_c_pde_N(N,k,:),nx,1));
        eps_rel_c_pde_N(N,k) = norm(reshape(dus_rel_c_pde_N(N,k,:),nx,1), lp);

        eps_d_pde(k) = norm(reshape(dus_d_pde(k,:),nx,1));
        eps_rel_d_pde(k) = norm(reshape(dus_rel_d_pde(k,:),nx,1), lp);

        eps_d_e(k) = norm(reshape(dus_d_e(k,:),nx,1));
        eps_rel_d_e(k) = norm(reshape(dus_rel_d_e(k,:),nx,1), lp);
    end
end


%% Plot errors
% Find indices for which we will plot the solution
i_plot = find(ts>=t_plot,1);
i_plot_pde = find(ts_pde>=t_plot,1);
i_start = ceil(i_plot*3/4);

figure(1);
clf;


% Plot Initial condition and solution at half nonlinear time
ax = subplot(2,2,1:2);
plot(xs_pde,us_pde(1,:),'k--','DisplayName',sprintf('Initial condition'));
hold on;
%plot(xs_pde,F0_fun(1,xs_pde),'k-.','DisplayName',sprintf('Source shape'));
%plot(xs,us_d(i_plot,:),'k-o','DisplayName',sprintf('Direct Euler solution at $T_{nl}/3$'));


for N = [Ns(N_max)]
    ax.ColorOrderIndex = N;
    plot(xs,reshape(ys_c_N(N, 100, 1:nx),nx,1),'-*','DisplayName',sprintf('Carleman solution at $n_t = 100$, $N=%d$',N));
end


%%%%%%%%%%

for N = [Ns(N_max)]
    ax.ColorOrderIndex = N+1;
    plot(xs,reshape(ys_c_N(N, nt*2/10, 1:nx),nx,1),'-*','DisplayName',sprintf('Carleman solution at $n_t =200$, $N=%d$',N));
end

%%%%%%%%%%%%%%%%
%%%%%%%%%%

for N = [Ns(N_max)]
    ax.ColorOrderIndex = N+2;
    plot(xs,reshape(ys_c_N(N, nt*5/10, 1:nx),nx,1),'-*','DisplayName',sprintf('Carleman solution at $n_t = 500$, $N=%d$',N));
end

%%%%%%%%%%%%%%%%

ylim([-max(abs(us_pde(1,:))), max(abs(us_pde(1,:)))]);

% Plot absolute l_2 error between Carleman and direct solution
for N = Ns
    ax = subplot(2,2,3);
    ax.ColorOrderIndex = N;
    semilogy(ts, eps_c_d_N(N,:),'DisplayName',sprintf('Taylor, $K=%d$',N));
    hold on;
end

% Plot time-maximum absolute l_2 error between Carleman and pdepe
ax = subplot(2,2,4);
semilogy(Ns, max(eps_c_d_N, [], 2),'-o','DisplayName',sprintf('Time-maximum error'));

% Format initial condition plot
subplot(2,2,1:2);
xlabel('$x$', 'interpreter','latex');
ylabel('$u(x, n_t)$', 'interpreter','latex');
xlim([x0 x1]);
lgd = legend();
set(lgd,'fontsize',fontsize-4);
set(gca,'fontsize',fontsize);

% Format absolute l_2 error plot
subplot(2,2,3);
title(sprintf('Absolute error'), 'interpreter','latex');
xlabel('$t$', 'interpreter','latex');
ylabel('$\|\varepsilon_{\mathrm{abs}}\|_{2}$', 'interpreter','latex');
xline(t_plot,':','DisplayName','T_{nl}/3', 'HandleVisibility', 'Off');
%5ylim([min([min(eps_c_d_N(:,i_start:end)),eps_d_pde(i_start:end)])*0.1 max(eps_c_d_N(1,:))*10]);
lgd = legend();
set(lgd,'fontsize',fontsize-4);
set(gca,'fontsize',fontsize);

ax = gca;
xruler = ax.XRuler;
old_fmt = xruler.TickLabelFormat;
old_xticks = xruler.TickValues;
old_labels = sprintfc(old_fmt, old_xticks);
new_tick = t_plot;
new_label = sprintf(['%s%d%s' old_fmt],'$T_{\mathrm{nl}}/3$');
all_xticks = [old_xticks, new_tick];
all_xlabels = [old_labels, new_label];
[new_xticks, sort_order] = sort(all_xticks);
new_labels = all_xlabels(sort_order);
%set(xruler, 'TickValues', new_xticks, 'TickLabels', new_labels);

% Format error convergence plot
subplot(2,2,4);
title(sprintf('Error convergence'), 'interpreter','latex');
xlabel('$K$', 'interpreter','latex');
ylabel('$\max_t \|\varepsilon_{\mathrm{abs}}\|_{2}$', 'interpreter','latex');
ax = gca;
lgd = legend();
set(gca,'fontsize',fontsize);
set(lgd,'fontsize',fontsize-4);

% Finalize and save
Re_act = max(max(us_pde))*L0/nu;
sgtitle(sprintf('Fisher-KPP-solution with, $(n_x)_{max}=%d$, $(n_t)_{max}=%d$, $\\mathrm{R}=%.2f$', nx, nt, R), 'interpreter','latex', 'fontsize', fontsize+2);
%savefig(sprintf('rde_re0_%.2f_N_%d_nx_%d_nt_%d_rev2.fig',Re0,N_max,nx,nt));

figure(2)
temp = ys_c_N(3, 1:nt, 1:nx);
temp1 = squeeze(temp);
mesh(temp1);


title(sprintf('Solution 3D plot'), 'interpreter','latex', 'FontSize', fontsize);
xlabel('$\textbf{$n_x$}$', 'interpreter','latex', 'FontSize', fontsize);
ylabel('$\textbf{$n_t$}$', 'interpreter','latex', 'FontSize', fontsize);
zlabel('$u(x,t)$', 'interpreter','latex', 'FontSize', fontsize);

