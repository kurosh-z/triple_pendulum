function [p0, p1, p2, p3, v0, v1, v2, v3]=tripleInvPenKinematics(z)
% cpmputes the Kinematics of triple pendulum 
% z=[q0, q1, q2, q3, q0d, q1d, q2d, q3d]

q0=z(1,:);
q1=z(2, :);
q2=z(3, :);
q3=z(4, :);

q0d=z(5, :);
q1d=z(6, :);
q2d=z(7, :);
q3d=z(8, :);

empty=zeros(size(q0)); % thats a trick to solve the vectorization of constants in matlabFunction

[p0, p1, p2, p3, v0, v1, v2, v3]=autoGen_3InvPenKinematics(q0,q1,q2,q3,q0d,q1d,q2d,q3d,empty);

end
