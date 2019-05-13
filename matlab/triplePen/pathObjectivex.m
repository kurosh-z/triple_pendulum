function obj=pathObjectivex(x,u)

Q=diag([1e5,1,1,1,10,1,1,1]);

nStates=size(x,1);
nSteps=size(x,2);

xCost=zeros(1,nSteps);
for i=1:nSteps
    xi=x((i-1)*nStates+1:i*nStates);
    xCost(i)=xi*Q*xi';
end

R=10;
uCost=R.*(u.^2);

obj=uCost+xCost;
% obj = R.*(u.^2);
if nargout == 2  % Analytic gradients
%     nTime = length(u);
    
    objGrad = [];
  
    
end



end