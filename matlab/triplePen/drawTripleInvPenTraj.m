function drawTripleInvPenTraj(t,p0,p1,p2,p3,nFrame)
% drawCartPoleTraj(t,p1,p2,nFrame)
%
% INPUTS:
%   t =  [1,n] = time stamp for the data in p1 and p2
%   p1 = [2,n] = [x;y] = position of center of the cart
%   p2 = [2,n] = [x;y] = position of tip of the pendulum
%   nFrame = scalar integer = number of "freeze" frames to display
%

clf; hold on;

Cart_Width = 0.08;
Cart_Height = 0.06;

Pen_Width = 2.5;  %pixels

%%%% Figure out the window size:

[xLow, xUpp, yLow, yUpp] = getBounds(p0,p1,p2,p3);

xLow = xLow - 0.7*Cart_Width;
xUpp = xUpp + 0.7*Cart_Width;

yLow = yLow - 0.7*Cart_Height;
yUpp = yUpp + 0.7*Cart_Height;

Limits = [xLow,xUpp,yLow,yUpp];

%%%% Get color map for the figure
map = colormap(parula);
tMap = linspace(t(1),t(end),size(map,1))';

%%%% Plot Rails
plot([Limits(1) Limits(2)],-0.5*Cart_Height*[1,1],'k-','LineWidth',4)

%%%% Draw the trace of the pendulum tip  (continuously vary color)

nTime = length(t);
for i=1:(nTime-1)
    idx = i:(i+1);
    x = p3(1,idx);
    y = p3(2,idx);
    c = interp1(tMap,map,mean(t(idx)));
    plot(x,y,'Color',c, 'LineWidth',1);
end

%%%% Compute the frames for plotting:
tFrame = linspace(t(1), t(end), nFrame);
cart = interp1(t',p0',tFrame')';
pen1 = interp1(t',p1',tFrame')';
pen2 = interp1(t',p2',tFrame')';
pen3 = interp1(t',p3',tFrame')';


for i = 1:nFrame
    
    % Compute color:
    color = interp1(tMap,map,tFrame(i));
    if i == nFrame
        color=interp1(tMap,map,tFrame(1));
    end    
        
    %Plot Cart
    x = cart(1,i) - 0.5*Cart_Width;
    y = -0.5*Cart_Height;
    w = Cart_Width;
    h = Cart_Height;
    hCart = rectangle('Position',[x,y,w,h],'LineWidth',2);
    set(hCart,'FaceColor',0.9*color);
    set(hCart,'EdgeColor',0.6*color);
    
    %Plot Pendulum1
    Rod_X = [cart(1,i), pen1(1,i)];
    Rod_Y = [cart(2,i), pen1(2,i)];
    plot(Rod_X,Rod_Y,'k-','LineWidth',Pen_Width,'Color',color)
    
    %Plot Bob and hinge
    plot(pen1(1,i),pen1(2,i),'k.','MarkerSize',20,'Color',color)
    plot(cart(1,i),cart(2,i),'k.','MarkerSize',20,'Color',color)
    
    %Plot Pendulum2
    Rod_X = [pen1(1,i), pen2(1,i)];
    Rod_Y = [pen1(2,i), pen2(2,i)];
    plot(Rod_X,Rod_Y,'k-','LineWidth',Pen_Width,'Color',0.3*color)
    %Plot Bob and hinge
    plot(pen2(1,i),pen2(2,i),'k.','MarkerSize',20,'Color',0.3*color)
    
    %Plot Pendulum3
    Rod_X = [pen2(1,i), pen3(1,i)];
    Rod_Y = [pen2(2,i), pen3(2,i)];
    plot(Rod_X,Rod_Y,'k-','LineWidth',Pen_Width,'Color',0.7*color)
    %Plot Bob and hinge
    plot(pen3(1,i),pen3(2,i),'k.','MarkerSize',20,'Color',0.7*color)
end

%These commands keep the window from automatically rescaling in funny ways.
axis(Limits);
% axis('equal');
axis manual;
% axis off;
grid on;
grid minor;
% xticks([-2 -1.5 -1 -.9 -.8 -.7 -.6 -.5 0 .5  .6 .7 .8 .9 1 1.2 1.5 2])
xticks([ -1.5 -1  -.5 0 .2  .5  1  1.5])
end



function [xLow, xUpp, yLow, yUpp] = getBounds(p0,p1,p2,p3)
%
% Returns the upper and lower bound on the data in val
%

val = [p0,p1,p2,p3];
xLow = min(val(1,:));
xUpp = max(val(1,:));
yLow = min(val(2,:));
yUpp = max(val(2,:));

end