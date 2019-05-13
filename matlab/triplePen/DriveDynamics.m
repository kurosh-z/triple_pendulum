
clear all;
clc;
syms t l1 l2 l3  m0 m1 m2 m3 a1 a2 a3 J1 J2 J3 d1 d2 d3 g  real
syms qq0(t) qq1(t) qq2(t) qq3(t) 

% syms q0 q1 q0_dot q1_dot q1_dd real 

% to make typing easy !
c1= cos(qq1(t)); s1= sin(qq1(t));
c2= cos(qq2(t)); s2= sin(qq2(t));
c3= cos(qq3(t)); s3= sin(qq3(t));

% defining omegas :
omega1= [0 0 diff(qq1(t), t)];
omega2= [0 0 diff(qq2(t), t)];
omega3= [0 0 diff(qq3(t), t)];

% defining mass centers position  :
p0=[qq0(t), 0];
p1=[qq0(t)-a1*s1, a1*c1];
p2=[qq0(t)-l1*s1 - a2*s2, l1*c1+a2*c2];
p3=[qq0(t)-l1*s1 - l2*s2 - a3*s3, l1*c1 + l2*c2 + a3*c3];

% calculate velocities :
v0=diff(p0, t);
v1=diff(p1, t);
v2=diff(p2, t);
v3=diff(p3, t);

% ready to calculate kinetic and potential energy:
T0=0.5*m0*v0*(v0.');
T1=0.5*m1*v1*(v1.') + 0.5*J1*omega1*(omega1.');
T2=0.5*m2*v2*(v2.') + 0.5*J2*omega2*(omega2.');
T3=0.5*m3*v3*(v3.') + 0.5*J3*omega3*(omega3.');

T=expand(T0+T1+T2+T3);

U=g*(m1*p1(2)+m2*p2(2)+m3*p3(2));

% Rayleigh dissipation function
R=0.5*d1*omega1(3)^2 + 0.5*d2*(omega2(3)-omega1(3))^2 + 0.5*d3*(omega3(3)-omega2(3))^2;


L=T-U;


% calculating derivatives wrt qdots
Ldq0= deriv(L, diff(qq0(t), t));
Ldq1= deriv(L, diff(qq1(t), t));
Ldq2= deriv(L, diff(qq2(t), t));
Ldq3= deriv(L, diff(qq3(t), t));

% calculating derivatives wrt time
Lt0=diff(Ldq0, t);
Lt1=diff(Ldq1, t);
Lt2=diff(Ldq2, t);
Lt3=diff(Ldq3, t);

% calculating derivatives wrt qs
Lq0= deriv(L, qq0(t));
Lq1= deriv(L, qq1(t));
Lq2= deriv(L, qq2(t));
Lq3= deriv(L, qq3(t));

% calculating R wrt qds
Rdq0=deriv(R, diff(qq0(t), t));
Rdq1=deriv(R, diff(qq1(t), t));
Rdq2=deriv(R, diff(qq2(t), t));
Rdq3=deriv(R, diff(qq3(t), t));

% calculating the LHS of Lagrange equations:
LHS0= Lt0 - Lq0 +Rdq0;
LHS1= Lt1 - Lq1 +Rdq1;
LHS2= Lt2 - Lq2 +Rdq2;
LHS3= Lt3 - Lq3 +Rdq3;

% substituting qdd0 with u and simplifying the results:
eq0=expr_simplifyer(LHS0);
eq1=expr_simplifyer(LHS1);
eq2=expr_simplifyer(LHS2);
eq3=expr_simplifyer(LHS3);

% defining all equations as a matrix
syms q0 q1 q2 q3 q0d q1d q2d q3d q1dd q2dd q3dd u real
Eq=[eq0; eq1; eq2; eq3];

%find mass matrix
acc=[u, q1dd , q2dd, q3dd];
mass_matrix= jacobian(Eq, acc);

acc_list={u, q1dd , q2dd, q3dd};
forcing_vector=-subs(Eq,acc_list,{0, 0, 0, 0});

mass_matrix33=mass_matrix(2:end,2:end);
forcing_vec33= forcing_vector(2:end);

% partial linearization for output q0dd
syms Det
mass_matrix33_det=det(mass_matrix33);
mass_matrix33_inv=adjoint(mass_matrix33)/Det;
phi_dd=mass_matrix33_inv *(-u *mass_matrix(2:end,1) + forcing_vec33);
phi_dd=simplify(subs(phi_dd, Det, mass_matrix33_det));


% % substitue values for parameters:
param_list={l1, l2, l3, a1, a2, a3, m0, m1, m2, m3, J1, J2, J3, d1, d2, d3, g};
param_values={0.32,0.419,0.485,0.20001517,0.26890449,0.21666087,3.34,0.8512,...
    0.8973,0.5519,0.01980194,0.02105375,0.01818537,0.00715294,1.9497e-06,...
    0.00164642,9.81};
% 
phi_dd_val=simplify(subs(phi_dd, param_list, param_values));

% % Phidd_t=vpa(subs(phi_dd_val,qss,val))


% defining ff= x_dot :
ff=[q0d; q1d; q2d; q3d; u; phi_dd_val(1);phi_dd_val(2);phi_dd_val(3)];

% convert ff to matlab function 
matlabFunction(ff(5), ff(6), ff(7), ff(8),...
    'file', 'autoGen_3InvPenDynamics.m',...
    'vars', {q1, q2, q3,q1d, q2d, q3d, u},...   % phi_dd dosen't depend on q0 and q0d so we don't need to 
    'outputs', {'uu', 'q1dd', 'q2dd', 'q3dd'}); % include them in vars !


%matlab function for kinematics:
% defining positions and velocities for cart and tip of each pendel  :
p00=[qq0(t); 0];
p11=[qq0(t)-l1*s1; l1*c1];
p22=[qq0(t)-l1*s1 - l2*s2; l1*c1+l2*c2];
p33=[qq0(t)-l1*s1 - l2*s2 - l3*s3; l1*c1 + l2*c2 + l3*c3];

% calculate velocities :
v00=diff(p00, t);
v11=diff(p11, t);
v22=diff(p22, t);
v33=diff(p33, t);

% substitue parameters and simplifying :
p00=subs(expr_simplifyer(p00), param_list, param_values);
p11=subs(expr_simplifyer(p11), param_list, param_values);
p22=subs(expr_simplifyer(p22), param_list, param_values);
p33=subs(expr_simplifyer(p33), param_list, param_values);

v00=subs(expr_simplifyer(v00), param_list, param_values);
v11=subs(expr_simplifyer(v11), param_list, param_values);
v22=subs(expr_simplifyer(v22), param_list, param_values);
v33=subs(expr_simplifyer(v33), param_list, param_values);

syms empty 'real' %fixes a bug in matlabFunction related to vectorization of constant terms
p00(2)=p00(2)+empty;
v00(2)=v00(2)+empty;


matlabFunction(p00, p11, p22, p33,...
    v00, v11, v22, v33,...
    'file','autoGen_3InvPenKinematics.m',...
    'vars',{q0, q1, q2, q3, q0d, q1d, q2d, q3d, empty},...
    'outputs',{'p00','p11','p22','p33','v00','v11', 'v22', 'v33'});

disp('system modeling is finished results are saved as functions')
function fout =deriv(f, g)
% derive differenctiates f with resprect to g=g(t)
% the variabel g=g(t) is a fuction of time
syms t x dx
lg={diff(g,t), g};
lx={dx, x};
f1= subs(f, lg, lx);
f2= diff(f1, x);
fout= subs(f2, lx, lg);
end


function expr_simplified= expr_simplifyer(expr)
% substitutes q(t) with q and derivatives with 
% qd, qdd and q0dd with u!

syms t qq0(t) qq1(t) qq2(t) qq3(t)
qts0={qq0(t), diff(qq0(t), t), diff(qq0(t), t, 2)};
qts1={qq1(t), diff(qq1(t), t), diff(qq1(t), t, 2)};
qts2={qq2(t), diff(qq2(t), t), diff(qq2(t), t, 2)};
qts3={qq3(t), diff(qq3(t), t), diff(qq3(t), t, 2)};

syms q0 q1 q2 q3 q0d q1d q2d q3d u q1dd q2dd q3dd real 
qs0= {q0, q0d, u};
qs1= {q1, q1d, q1dd};
qs2= {q2, q2d, q2dd};
qs3= {q3, q3d, q3dd};

expr0=subs(expr , qts0, qs0);
expr1=subs(expr0, qts1, qs1);
expr2=subs(expr1, qts2, qs2);
expr3=subs(expr2, qts3, qs3);

expr_simplified=simplify(expr3);

end









