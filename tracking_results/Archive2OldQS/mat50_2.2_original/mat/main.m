

trfName='npContent.mat';
myVars={'time', 'state', 'control'};
Temp=load(trfName, myVars{:});
nTime=size(Temp.time,2);
nState=size(Temp.state,1)*size(Temp.state,2)/nTime;
T.time=Temp.time;
T.state=reshape(Temp.state,nState, nTime);
T.control=reshape(Temp.control, 1, nTime);

q0= T.state(1,:);
q1= T.state(2,:);
q2= T.state(3,:);
q3= T.state(4,:);


dq0= T.state(5,:);
dq1= T.state(6,:);
dq2= T.state(7,:);
dq3= T.state(8,:);

u = T.control;

% draw trajecotry
t = T.time;
[p0, p1, p2, p3, v0, v1, v2, v3] = tripleInvPenKinematics(T.state);

% figure(1); clf;
% nFrame = 15;  %Number of frames to draw
% drawTripleInvPenTraj(t,p0,p1,p2,p3,nFrame);

% nFrame=40;
% figure(2); clf 
% drawTimeShift(t,p0,p1,p2,p3,nFrame);

figure(3);clf;
nFrameAnim=120;
drawTripleInvPenAnim(t,p0,p1,p2,p3,q1,q2,q3,nFrameAnim,'AnimTracking.avi')


% % Plot the solution:
% figure(3); clf;
% 
% subplot(2,1,1)
% plot(t,q1*180/pi)
% ylabel('q [degree]','color','b')
% title('Triple Pendulum Swing-Up');
% hold on
% plot(t,q2*180/pi,'color','r')
% hold on
% plot(t,q3*180/pi,'color','g')
% legend('q1', 'q2', 'q3')
% grid on;
% grid minor;
% 
% 
% subplot(2,1,2)
% plot(t,dq1,'-o','DisplayName','q1d')
% ylabel('qdot[rad/s]')
% hold on
% plot(t,dq2, '-o','DisplayName','q2d')
% hold on
% plot(t,dq3, '-o','DisplayName','q3d')
% 
% 
% figure(4); clf;
% subplot(3,1,1)
% plot(t,q0, '-o', 'DisplayName','q0')
% ylabel('q0 [m]')
% title('Triple Pendulum Swing-Up');
% 
% subplot(3,1,2)
% plot(t,dq0, '-o', 'DisplayName','q0d')
% ylabel('q0d [m/s]')
% 
% subplot(3,1,3)
% plot(t,u, '-o', 'DisplayName', 'u')
% ylabel('u [m/s^2')
% 
% 
% 
