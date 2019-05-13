function [tSim, xSim]= OpenloopSimulation(soln, t, x0, anim)


%[tSim,xSim] = ode45(@(t,x)(tripelInvPenDynamics(x,soln(end).interp.control(t))),t,x0);

penSim=@(t,x)(tripelInvPenDynamics(x,soln(end).interp.control(t)));
[tSim,xSim] = ode45(penSim,t,x0);

q0Sim= xSim(:,1);
q1Sim= xSim(:,2);
q2Sim= xSim(:,3);
q3Sim= xSim(:,4);
qdot0Sim=xSim(:,5);
qdot1Sim=xSim(:,6);
qdot2Sim=xSim(:,7);
qdot3Sim=xSim(:,8);

zInterp=soln(end).interp.state(tSim');
q0Interp= zInterp(1,:);
q1Interp= zInterp(2,:);
q2Interp= zInterp(3,:);
q3Interp= zInterp(4,:);
qdot0Interp=zInterp(5,:);
qdot1Interp=zInterp(6,:);
qdot2Interp=zInterp(7,:);
qdot3Interp=zInterp(8,:);

deltaq0= abs(q0Sim' - q0Interp);
deltaq1= abs(q1Sim' - q1Interp);
deltaq2= abs(q2Sim' - q2Interp);
deltaq3= abs(q3Sim' - q3Interp);
deltaqdot0= abs(qdot0Sim' - qdot0Interp);
deltaqdot1= abs(qdot1Sim' - qdot1Interp);
deltaqdot2= abs(qdot2Sim' - qdot2Interp);
deltaqdot3= abs(qdot3Sim' - qdot3Interp);

%% Plot the solution:
figSim=figure(33); clf;

left_color = [0 0 0];
right_color = hex2rgb('#0000ff');
set(figSim,'defaultAxesColorOrder',[left_color; right_color]);
figSim.PaperUnits='centimeters';  
figSim.PaperPosition = [ 0 0 16 11];
set(figSim,'PaperSize',[16 11])
subplot(2,1,1)
yyaxis left
% first q0:
plot(tSim, q1Sim*180/pi, 'color', hex2rgb('#ff7f0e'),'LineWidth',2)
hold on
plot(tSim, q1Interp*180/pi,'--', 'color', hex2rgb('#ff7f0e'),'LineWidth',1)
hold on

yyaxis left
plot(tSim, q2Sim*180/pi,'-', 'color', hex2rgb('#1f77b4'),'LineWidth',2)
hold on
plot(tSim, q2Interp*180/pi,'--', 'color', hex2rgb('#1f77b4'),'LineWidth',1)

yyaxis left
plot(tSim, q3Sim*180/pi,'-', 'color', hex2rgb('#2ca02c'),'LineWidth',2)
plot(tSim, q3Interp*180/pi,'--', 'color', hex2rgb('#2ca02c'),'LineWidth',1)

ylabel('$q_i$ in $\circ$,  $_{i=1,2,3}$','Interpreter','latex','FontSize',12)

xlim([0 tSim(end)]) 
grid on;

% right axis :
yyaxis right
plot(tSim,q0Sim ,'-', 'color', hex2rgb('#0000ff'),'LineWidth',1), hold on 
plot(tSim, q0Interp,'--', 'color', hex2rgb('#0000ff'),'LineWidth',1)
ylabel('$q_0$,  in $m$','Interpreter','latex','FontSize',12)
lgnd3=legend({'$q_1-Sim$','$q_1-Interp$','$q_2-Sim$','$q_2-Interp$','$q_3-Sim$','$q_3-Interp$','$q_0-Sim$','$q_0-Interp$'},'Interpreter','latex','Location','southwest');
set(lgnd3.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.95;.95;.95;.6]));

% angular velocities:
subplot(2,1,2)
yyaxis left
plot(tSim,qdot1Sim*180/pi,'-','color',hex2rgb('#ff7f0e'),'LineWidth',2), hold on
plot(tSim, qdot1Interp*180/pi,'--', 'color', hex2rgb('#ff7f0e'),'LineWidth',1)
ylabel('$\dot{q}_i$, in $\circ$/$s$,  $_{i=1,2,3}$','Interpreter','latex','FontSize',12)
hold on
plot(tSim,qdot2Sim*180/pi,'-','color',hex2rgb('#1f77b4'),'LineWidth',2), hold on 
plot(tSim, qdot2Interp*180/pi,'--', 'color', hex2rgb('#1f77b4'),'LineWidth',1)
hold on
plot(tSim,qdot3Sim*180/pi,'-','color',  hex2rgb('#2ca02c'),'LineWidth',2), hold on 
plot(tSim, qdot3Interp*180/pi,'--', 'color', hex2rgb('#2ca02c'),'LineWidth',1)
xlim([0 tSim(end)]) ;
grid on;
%right axis , qdot_0
yyaxis right
plot(tSim,qdot0Sim,'-','color',hex2rgb('#0000ff') ,'Linewidth',1), hold on 
plot(tSim, qdot0Interp,'--', 'color', hex2rgb('#0000ff'),'LineWidth',1)

ylabel('$\dot{q}_0$,  in $m/s$','Interpreter','latex','FontSize',12)
lgnd32=legend({'$\dot{q}_1-Sim$','$\dot{q}_1-Interp$','$\dot{q}_2-Sim$','$\dot{q}_2-Interp$','$\dot{q}_3-Sim$','$\dot{q}_3-Interp$','$\dot{q}_0-Sim$','$\dot{q}_3-Interp$',},'Interpreter','latex','Location','southwest');
set(lgnd32.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.95;.95;.95;.6]));
xlabel('$t\hspace{.1cm}in\hspace{.1cm}s$','Interpreter','latex','FontSize',12)
print(figSim,'figSimOl','-dpdf')

%% plot delta x
figdeltaSim=figure(333); clf;
set(figdeltaSim,'defaultAxesColorOrder',[left_color; right_color]);
figdeltaSim.PaperUnits='centimeters';  
figdeltaSim.PaperPosition = [ 0 0 16 9];
set(figdeltaSim,'PaperSize',[16 9])
subplot(2,1,1)
yyaxis left
% first q0:
plot(tSim, deltaq1*180/pi, 'color', hex2rgb('#ff7f0e'),'LineWidth',2)
hold on

yyaxis left
plot(tSim, deltaq2*180/pi,'-', 'color', hex2rgb('#1f77b4'),'LineWidth',2)


yyaxis left
plot(tSim, deltaq3*180/pi,'-', 'color', hex2rgb('#2ca02c'),'LineWidth',2)


ylabel('$\Delta q_i$ in $\circ$,  $_{i=1,2,3}$','Interpreter','latex','FontSize',12)

xlim([0 tSim(end)]) 
grid on;

% right axis :
yyaxis right
plot(tSim,deltaq0 ,'-', 'color', hex2rgb('#0000ff'),'LineWidth',1)

ylabel('$\Delta q_0$,  in $m$','Interpreter','latex','FontSize',12)
lgnd3=legend({'$\Delta q_1$','$\Delta q_2$','$\Delta q_3$','$\Delta q_0$'},'Interpreter','latex','Location','northwest');
set(lgnd3.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.95;.95;.95;.6]));

% angular velocities:
subplot(2,1,2)
yyaxis left
plot(tSim,deltaqdot1*180/pi,'-','color',hex2rgb('#ff7f0e'),'LineWidth',2)
ylabel('$\Delta \dot{q}_i$, in $\circ$/$s$,  $_{i=1,2,3}$','Interpreter','latex','FontSize',12)
hold on
plot(tSim,deltaqdot2*180/pi,'-','color',hex2rgb('#1f77b4'),'LineWidth',2)
hold on
plot(tSim,deltaqdot3*180/pi,'-','color',  hex2rgb('#2ca02c'),'LineWidth',2)
xlim([0 tSim(end)]) ;
grid on;
%right axis , qdot_0
yyaxis right
plot(tSim,deltaqdot0,'-','color',hex2rgb('#0000ff') ,'Linewidth',1)

ylabel('$\Delta \dot{q}_0$,  in $m/s$','Interpreter','latex','FontSize',12)
lgnd32=legend({'$\Delta \dot{q}_1$','$\Delta \dot{q}_2$','$\Delta \dot{q}_3$','$\Delta \dot{q}_0$'},'Interpreter','latex','Location','northwest');
set(lgnd32.BoxFace, 'ColorType','truecoloralpha', 'ColorData',uint8(255*[.95;.95;.95;.6]));
xlabel('$t\hspace{.1cm}in\hspace{.1cm}s$','Interpreter','latex','FontSize',12)
print(figdeltaSim,'figdeltaSimOl','-dpdf')

%% animation
% if anim == 'True'
%     
%     
%     [p0, p1, p2, p3, v0, v1, v2, v3] = tripleInvPenKinematics(xSim');
%     
%     fig11= figure(1); clf;
%     set(fig11,'defaultAxesColorOrder',[left_color; right_color]);
%     fig11.PaperUnits='centimeters';
%     fig11.PaperPosition = [ 0 0 12 10];
%     set(fig11,'PaperSize',[12 10])
%     nFrame = 14;  %Number of frames to draw
%     drawTripleInvPenTraj(t,p0,p1,p2,p3,nFrame);
%     print(fig11,'fig1OL','-dpdf')
%     
%     
% %     fig222=figure(22);
% %     set(fig222,'defaultAxesColorOrder',[left_color; right_color]);
% %     fig222.PaperUnits='centimeters';
% %     fig222.PaperPosition = [ 0 0 12 10];
% %     set(fig222,'PaperSize',[12 10])
% %     nShiftFrames=25;
% %     drawTimeShift(t,p0,p1,p2,p3,nShiftFrames)
% %     print(fig222,'fig22OL','-dpdf')
%      
%     figure(29);clf;
%     nFrameAnim=120;
%     drawTripleInvPenAnim(t,p0,p1,p2,p3,q1Sim,q2Sim,q3Sim,nFrameAnim, 'TripPenOL.avi')
%     
    
% end
end