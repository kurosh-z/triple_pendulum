% MAIN - Pendulum
%
%

clc; clear;
addpath ../OptimTraj

maxDist=1;  % maximal distance allowed for each direction
maxacc = 40;  %Maximum acceleration of cart (our input u)
duration = 2.1;   % duration of swing_up

% defining Equlibrium points :
eq_ttt=[0; 0; 0; 0; 0; 0; 0; 0];
eq_bbb=[0; pi; pi; pi; 0; 0; 0; 0];

% User-defined dynamics and objective functions
problem.func.dynamics = @(t,x,u)( tripelInvPenDynamics(x,u) );
problem.func.pathObj = @(t,x,u)( pathObjective(x,u)); % min 
% problem.func.pathObj = @(t,x,u)( pathObjectivex(x,u) ); % min q0
%problem.func.pathObj = @(t,x,u)(ones(size(t)));  % min time!
%problem.func.pathObj = @(t,x,u)(pathObjectiveMint(x,u,t) );  % min time!
% Problem bounds
problem.bounds.initialTime.low = 0;
problem.bounds.initialTime.upp = 0;
% problem.bounds.finalTime.low = duration-0.6;
% problem.bounds.finalTime.upp = duration+0.3;

problem.bounds.finalTime.low =duration-.1;
problem.bounds.finalTime.upp =duration+.1;

eps=[1e-6;1e-6;1e-6;1e-6;1e-6;1e-6;1e-6;1e-6];
problem.bounds.initialState.low = eq_bbb-eps;
problem.bounds.initialState.upp = eq_bbb+eps;

problem.bounds.finalState.low = eq_ttt-eps;
problem.bounds.finalState.upp = eq_ttt+eps;
problem.bounds.state.low = [-2*maxDist; -2*pi;-2*pi;-2*pi;-5; -inf;-inf; -inf];
problem.bounds.state.upp = [ 2*maxDist;  2*pi; 2*pi; 2*pi; 5; inf; inf;  inf];

problem.bounds.control.low = -maxacc; %-inf;
problem.bounds.control.upp = maxacc; %inf;

% Guess at the initial trajectory
problem.guess.time = [0,duration];
problem.guess.state = [eq_bbb, eq_ttt];
problem.guess.control = [0, 0];


%%%% Switch between a variety of methods

% method = 'trapezoid';
% method = 'trapGrad';   
method = 'hermiteSimpson';
% method = 'hermiteSimpsonGrad';   
% method = 'chebyshev';   
% method = 'rungeKutta';  
% method = 'rungeKuttaGrad';
% method = 'gpops';



%%%% Method-independent options:
problem.options(1).nlpOpt = optimset(...
    'Display','iter',...   % {'iter','final','off'}
    'TolFun',1e-3,...
    'MaxFunEvals',1e8,...
    'TolCon',1e-7,...
    'TolX',1e-10);   %options for fmincon

problem.options(2).nlpOpt = optimset(...
    'Display','iter',...   % {'iter','final','off'}
    'TolFun',1e-4,...
    'MaxFunEvals',5e8,...
    'TolCon',1e-10,...
    'TolX',1e-12);   %options for fmincon

% problem.options(3).nlpOpt = optimset(...
%     'Display','iter',...   % {'iter','final','off'}
%     'TolFun',1e-4,...
%     'MaxFunEvals',5e8,...
%     'TolCon',1e-15,...
%     'TolX',1e-12);   %options for fmincon

switch method
    
    case 'trapezoid'
        problem.options(1).method = 'trapezoid'; % Select the transcription method
        problem.options(1).trapezoid.nGrid = 10;  %method-specific options
        
        problem.options(2).method = 'trapezoid'; % Select the transcription method
        problem.options(2).trapezoid.nGrid = 25;  %method-specific options
        
    case 'trapGrad'  %trapezoid with analytic gradients
        
        problem.options(1).method = 'trapezoid'; % Select the transcription method
        problem.options(1).trapezoid.nGrid = 10;  %method-specific options
        problem.options(1).nlpOpt.GradConstr = 'on';
        problem.options(1).nlpOpt.GradObj = 'on';
        problem.options(1).nlpOpt.DerivativeCheck = 'off';
        
        problem.options(2).method = 'trapezoid'; % Select the transcription method
        problem.options(2).trapezoid.nGrid = 45;  %method-specific options
        problem.options(2).nlpOpt.GradConstr = 'on';
        problem.options(2).nlpOpt.GradObj = 'on';
        
    case 'hermiteSimpson'
        
        % First iteration: get a more reasonable guess
        problem.options(1).method = 'hermiteSimpson'; % Select the transcription method
        problem.options(1).hermiteSimpson.nSegment = 20;  %method-specific options
        
        % Second iteration: refine guess to get precise soln
        problem.options(2).method = 'hermiteSimpson'; % Select the transcription method
        problem.options(2).hermiteSimpson.nSegment = 50;  %method-specific options
        
%         problem.options(3).method = 'hermiteSimpson'; % Select the transcription method
%         problem.options(3).hermiteSimpson.nSegment = 50;  %method-specific options
        
    case 'hermiteSimpsonGrad'  %hermite simpson with analytic gradients
        
        problem.options(1).method = 'hermiteSimpson'; % Select the transcription method
        problem.options(1).hermiteSimpson.nSegment = 35;  %method-specific options
        problem.options(1).nlpOpt.GradConstr = 'on';
        problem.options(1).nlpOpt.GradObj = 'on';
        problem.options(1).nlpOpt.DerivativeCheck = 'off';
        
        problem.options(2).method = 'hermiteSimpson'; % Select the transcription method
        problem.options(2).hermiteSimpson.nSegment = 55;  %method-specific options
        problem.options(2).nlpOpt.GradConstr = 'on';
        problem.options(2).nlpOpt.GradObj = 'on';
        
        
    case 'chebyshev'
        
        % First iteration: get a more reasonable guess
        problem.options(1).method = 'chebyshev'; % Select the transcription method
        problem.options(1).chebyshev.nColPts = 9;  %method-specific options
        
        % Second iteration: refine guess to get precise soln
        problem.options(2).method = 'chebyshev'; % Select the transcription method
        problem.options(2).chebyshev.nColPts = 15;  %method-specific options
        
    case 'multiCheb'
        
        % First iteration: get a more reasonable guess
        problem.options(1).method = 'multiCheb'; % Select the transcription method
        problem.options(1).multiCheb.nColPts = 6;  %method-specific options
        problem.options(1).multiCheb.nSegment = 4;  %method-specific options
        
        % Second iteration: refine guess to get precise soln
        problem.options(2).method = 'multiCheb'; % Select the transcription method
        problem.options(2).multiCheb.nColPts = 9;  %method-specific options
        problem.options(2).multiCheb.nSegment = 4;  %method-specific options
        
    case 'rungeKutta'
        problem.options(1).method = 'rungeKutta'; % Select the transcription method
        problem.options(1).defaultAccuracy = 'low';
        problem.options(2).method = 'rungeKutta'; % Select the transcription method
        problem.options(2).defaultAccuracy = 'medium';
    
    case 'rungeKuttaGrad'
      
        problem.options(1).method = 'rungeKutta'; % Select the transcription method
        problem.options(1).defaultAccuracy = 'low';
        problem.options(1).nlpOpt.GradConstr = 'on';
        problem.options(1).nlpOpt.GradObj = 'on';
        problem.options(1).nlpOpt.DerivativeCheck = 'off';
        
        problem.options(2).method = 'rungeKutta'; % Select the transcription method
        problem.options(2).defaultAccuracy = 'medium';
        problem.options(2).nlpOpt.GradConstr = 'on';
        problem.options(2).nlpOpt.GradObj = 'on';
        
    case 'gpops'
        problem.options = [];
        problem.options.method = 'gpops';
        problem.options.defaultAccuracy = 'high';
        problem.options.gpops.nlp.solver = 'snopt';  %Set to 'ipopt' if you have GPOPS but not SNOPT
        
    otherwise
        error('Invalid method!');
end




% Solve the problem
soln = optimTraj(problem);

% save the results:
filename='Mat50_2.2.mat';
saveResults(soln,filename);

%%
% t = soln(end).grid.time;

q0= soln(end).grid.state(1,:);
q1= soln(end).grid.state(2,:);
q2= soln(end).grid.state(3,:);
q3= soln(end).grid.state(4,:);


dq0= soln(end).grid.state(5,:);
dq1= soln(end).grid.state(6,:);
dq2= soln(end).grid.state(7,:);
dq3= soln(end).grid.state(8,:);

u = soln(end).grid.control;

% Iterpolated curves 
time=linspace(0, soln(end).grid.time(end), 260);
zInterp=soln(end).interp.state(time);
q0Interp= zInterp(1,:);
q1Interp= zInterp(2,:);
q2Interp= zInterp(3,:);
q3Interp= zInterp(4,:);
qdot0Interp=zInterp(5,:);
qdot1Interp=zInterp(6,:);
qdot2Interp=zInterp(7,:);
qdot3Interp=zInterp(8,:);
uInterp=soln(end).interp.control(time);

% draw trajecotry
t = linspace(soln(end).grid.time(1), soln(end).grid.time(end), size(q0,2));
z = soln(end).interp.state(t);

[p0, p1, p2, p3, v0, v1, v2, v3] = tripleInvPenKinematics(z);

left_color = [0 0 0];
right_color = hex2rgb('#0000ff');
fig1= figure(1); clf;
set(fig1,'defaultAxesColorOrder',[left_color; right_color]);
fig1.PaperUnits='centimeters';  
fig1.PaperPosition = [ 0 0 12 10];
set(fig1,'PaperSize',[12 10])
nFrame = 14;  %Number of frames to draw
drawTripleInvPenTraj(t,p0,p1,p2,p3,nFrame);
print(fig1,'fig1','-dpdf')


figure(2);clf;
nFrameAnim=120;
drawTripleInvPenAnim(t,p0,p1,p2,p3,q1,q2,q3,nFrameAnim, 'TripPen.avi')

fig22=figure(22);
set(fig22,'defaultAxesColorOrder',[left_color; right_color]);
fig22.PaperUnits='centimeters';  
fig22.PaperPosition = [ 0 0 12 10];
set(fig22,'PaperSize',[12 10])
nShiftFrames=25;
drawTimeShift(t,p0,p1,p2,p3,nShiftFrames)
print(fig22,'fig22','-dpdf')

% Plot the solution:
fig3=figure(3); clf;
set(fig3,'defaultAxesColorOrder',[left_color; right_color]);
fig3.PaperUnits='centimeters';  
fig3.PaperPosition = [ 0 0 16 12];
set(fig3,'PaperSize',[16 12])
subplot(2,1,1)
yyaxis left
%scatter(t,q1*180/pi, 'o','b'), hold on 
plot(time, q1Interp*180/pi, 'color', hex2rgb('#ff7f0e'),'LineWidth',2)
hold on
% scatter(t,q2*180/pi,'.','r') , hold on 
yyaxis left
plot(time, q2Interp*180/pi,'-', 'color', hex2rgb('#1f77b4'),'LineWidth',2)
hold on
% scatter(t,q3*180/pi,'.','g'), hold on 
yyaxis left
plot(time, q3Interp*180/pi,'-', 'color', hex2rgb('#2ca02c'),'LineWidth',2)
% legend({'$q_1$','$q_1-Interp$', '$q_2$','$q_2-Interp$', '$q_3$', '$q_3-Interp$'},'Interpreter','latex')
% lgnd3=legend({'$q_1-Interp$','$q_2-Interp$','$q_3-Interp$'},'Interpreter','latex');
% set(legend,'color','none');
% set(legend, 'Box', 'off');

% ylabel('$q \hspace{.2cm} in \hspace{.2cm} ^\circ$','Interpreter','latex')
ylabel('$q_i$ in $\circ$,  $_{i=1,2,3}$','Interpreter','latex','FontSize',12)
% title('Aufschwingungstrajektorie');
% xticks([0 .5 1 1.5 2 2.1]);
xlim([0 time(end)]) 
grid on;

% right axis :
yyaxis right
plot(time,q0Interp ,'-', 'color', hex2rgb('#0000ff'),'LineWidth',1)
ylabel('$q_0$,  in $m$','Interpreter','latex','FontSize',12)
lgnd3=legend({'$q_1-Interp$','$q_2-Interp$','$q_3-Interp$','$q_0-Interp$'},'Interpreter','latex','Location','southwest');
set(lgnd3.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.95;.95;.95;.6]));

% angular velocities:
subplot(2,1,2)
yyaxis left
plot(time,qdot1Interp*180/pi,'-','color',hex2rgb('#ff7f0e'),'LineWidth',2)
ylabel('$\dot{q}_i$, in $\circ$/$s$,  $_{i=1,2,3}$','Interpreter','latex','FontSize',12)
hold on
plot(time,qdot2Interp*180/pi,'-','color',hex2rgb('#1f77b4'),'LineWidth',2)
hold on
plot(time,qdot3Interp*180/pi,'-','color',  hex2rgb('#2ca02c'),'LineWidth',2)
xlim([0 time(end)]) ;
grid on;
%right axis , qdot_0
yyaxis right
plot(time,qdot0Interp,'-','color',hex2rgb('#0000ff') ,'Linewidth',1)
ylabel('$\dot{q}_0$,  in $m/s$','Interpreter','latex','FontSize',12)
lgnd32=legend({'$\dot{q}_1-Interp$','$\dot{q}_2-Interp$','$\dot{q}_3-Interp$','$\dot{q}_0-Interp$'},'Interpreter','latex','Location','southwest');
set(lgnd32.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.95;.95;.95;.6]));
xlabel('$t\hspace{.1cm}in\hspace{.1cm}s$','Interpreter','latex','FontSize',12)
print(fig3,'fig3','-dpdf')

str=zeros(1,size(t,2));
str(1:3)=[1 2 3];
for i=4:2:size(t,2)-1
    str(i:i+1)=[2 3];
end
fig4=figure(4); clf;
plot(time,uInterp,'-','Linewidth',2),hold on
%scatter(t,u,'r')
xlim([0 time(end)]) ;
fig4.PaperUnits='centimeters';  
fig4.PaperPosition = [ 0 0 16 6];
set(fig4,'PaperSize',[16 6])
% labelpoints(t,u,str)
ylabel('$u \hspace{.1cm} in \hspace{.1cm} m/s^2$','Interpreter','latex')
xlabel('$t\hspace{.1cm}in\hspace{.1cm}s$','Interpreter','latex','FontSize',12)
lgnd4=legend({'$u-Interp$'},'Interpreter','latex');
set(lgnd4.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.95;.95;.95;.6]));
grid on;
print(fig4,'fig4','-dpdf')


% Plot the sparsity pattern
if isfield(soln(1).info,'sparsityPattern')
   figure(5); clf;
   spy(soln(1).info.sparsityPattern.equalityConstraint);
   axis equal
   title('Sparsity pattern in equality constraints')
end

 %%%% Show the error in the collocation constraint between grid points:
% %
if strcmp(soln(end).problem.options.method,'trapezoid') || strcmp(soln(end).problem.options.method,'hermiteSimpson')
    % Then we can plot an estimate of the error along the trajectory
%     figure(5); clf;
    
    % NOTE: the following commands have only been implemented for the direct
    % collocation(trapezoid, hermiteSimpson) methods, and will not work for
    % chebyshev or rungeKutta methods.
    cc = soln(end).interp.collCst(t);
    
%     subplot(1,2,1);
%     plot(t,cc(1,:))
%     title('Kollokationsfehler:  dx/dt - f(x,u,t)')
%     ylabel('q0d cart speed')
%     
%     idx = 1:length(soln(end).info.error);
%     subplot(1,2,2); hold on;
%     plot(idx,soln(end).info.error(1,:),'ko');
%     title('State Error')
%     ylabel('q0 cart position')
    
    
    fig6=figure(6);
    fig6.PaperUnits='centimeters';  
    fig6.PaperPosition = [ 0 0 17 17];
    fig6_pos=fig6.PaperPosition;
    %set(fig6,'PaperSize',[18 18])
    fig6.PaperSize = [fig6_pos(3) fig6_pos(4)];
    subplot(4,2,1);
    plot(t,cc(2,:)*180/pi)
    xlabel('$t$ in $s$', 'Interpreter','latex','FontSize',12)
%     title('{\boldmath$\frac{\mathrm{d}x}{\mathrm{d}t}-f(x,u,t)$}','Interpreter','latex','FontSize',12)
    title('   Kollokationsfehler','FontSize',12)
    ylabel('$\dot{q}_1$-Fehler in $\circ$/$s$','Interpreter','latex','FontSize',12)
    xlim([0 t(end)])
    
    subplot(4,2,3);
    plot(t,cc(3,:)*180/pi)
    xlabel('$t$ in $s$', 'Interpreter','latex','FontSize',12)
    ylabel('$\dot{q}_2$-Fehler in $\circ$/$s$','Interpreter','latex','FontSize',12)
    xlim([0 t(end)])
    
   
    
    subplot(4,2,5);
    plot(t,cc(4,:)*180/pi)
    xlabel('$t$ in $s$', 'Interpreter','latex','FontSize',12)
    ylabel('$\dot{q}_3$-Fehler in $\circ$/$s$','Interpreter','latex','FontSize',12)
    xlim([0 t(end)])
    
    
    idx = 1:length(soln(end).info.error);
    subplot(4,2,2); hold on;
    scatter(idx,soln(end).info.error(2,:)*180/pi,15,'o');
    xlabel('$segmentindex$','Interpreter','latex','FontSize',12)
    title('  Zustandsabweichung','FontSize',12)
    ylabel('$q_1$-Fehler in $\circ$','Interpreter','latex','FontSize',12);
    xlim([0 idx(end)])
    subplot(4,2,4); hold on;
    scatter(idx,soln(end).info.error(3,:)*180/pi,15,'o');
    xlabel('$segmentindex$','Interpreter','latex','FontSize',12)
    ylabel('$q_2$-Fehler in $\circ$','Interpreter','latex','FontSize',12);
    xlim([0 idx(end)])
     subplot(4,2,6); hold on;
    scatter(idx,soln(end).info.error(4,:)*180/pi,15,'o');
    xlabel('$segmentindex$','Interpreter','latex','FontSize',12)
    ylabel('$q_3$-Fehler in $\circ$','Interpreter','latex','FontSize',12);
    xlim([0 idx(end)])
    subplot(4,2,7);
    plot(t,cc(1,:))
    ylabel('$\dot{q}_0$-Fehler in $m$/$s$','Interpreter','latex','FontSize',12)
    xlim([0 t(end)])
    xlabel('$t$ in $s$', 'Interpreter','latex','FontSize',12)
    idx = 1:length(soln(end).info.error);
    subplot(4,2,8); hold on;
    scatter(idx,soln(end).info.error(1,:),15,'o');
    ylabel('$q_0$-Fehler in $m$','Interpreter','latex','FontSize',12)
    xlabel('$segmentindex$','Interpreter','latex','FontSize',12)
    xlim([0 idx(end)])
   
    print(fig6,'fig6','-dpdf')
end
