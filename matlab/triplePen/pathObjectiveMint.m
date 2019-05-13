function [obj, objGrad]=pathObjectiveMint(x,u,t)

Q=diag([1,1,1,1,100,1,1,1]);

nStates=size(x,1);
nSteps=size(x,2);

xCost=zeros(1,nSteps);
for i=1:nSteps
    xi=x((i-1)*nStates+1:i*nStates);
    xCost(i)=xi*Q*xi';
end

R=1;
uCost=R.*(u.^2);
tCost= 100000*ones(size(t));
obj=uCost+xCost+tCost;
% obj = R.*(u.^2);
if nargout == 2  % Analytic gradients
%     nTime = length(u);
    
    objGrad = [];
  
    
end

end

% print(fig,'MySavedPlot','-dpng')