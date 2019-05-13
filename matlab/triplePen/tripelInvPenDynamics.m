function [dx, dxGrad]=tripelInvPenDynamics(z, u)
% TODO: write dxGrad
% z=[q0, q1, q2, q3, q0d, q1d, q2d, q3d]

q0=z(1,:);
q1=z(2, :);
q2=z(3, :);
q3=z(4, :);

q0d=z(5, :);
q1d=z(6, :);
q2d=z(7, :);
q3d=z(8, :);

[q0dd, q1dd, q2dd, q3dd]=autoGen_3InvPenDynamics(q1,q2,q3,q1d,q2d,q3d,u);

dx=[q0d; q1d; q2d; q3d; q0dd; q1dd; q2dd; q3dd];

if nargout == 2 %analytic gradient
    % TODO: calculating dxGrad
    dxGrad=[];
end    
    
end