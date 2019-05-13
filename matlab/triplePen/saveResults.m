function saveResults(soln,filename)
%
% This will save the Grid Points to be used later in 
% python for Tracking 

state= soln(end).grid.state;
nState= size(state,1);
nTime= size(state,2);
control=soln(end).grid.control;clc

nControl= size(control,1);

time= linspace(soln(end).grid.time(1), soln(end).grid.time(end), nTime);
S.time= time;
S.control=reshape(control,1,nTime*nControl);
S.state= reshape(state,1,nTime*nState);
save(filename,'-Struct','S')

end