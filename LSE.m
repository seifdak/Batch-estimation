load dataset1.mat


%%Retreive Data from dataset

l = evalin('base','l');
r = evalin('base','r');
t = evalin('base','t');
v = evalin('base','v');
x_true = evalin('base','x_true');
v_var = evalin('base','v_var');
r_var = evalin('base','r_var');

%%Preparation of matrices 

zero_matrix = zeros(0,12709);
T= numel(t);
n= numel(x_true);
e= sparse(n,T);
%A= speye(e);
A= sparse(e);
A_inv = inv(A);
C = sparse(e);
H = [A_inv; C];
%H = H(1,:);
H= H(2:end, 1:end);

%Perpare batch matrices

Q= ones(length(y),1)/v_var*0.1*0.1;
R= ones(length(y),1)/r_var;
W = [Q; R];
W= W(2:end, 2:end);

% Left_triangle = [Q; zero_matrix];
% Right_triangle = [zero_matrix; R];
% W= [Left_triangle; Right_triangle];
%W = spdiags(W);
% W = W(1,:);
% W = W(:,1);

W= W(2:end, 2:end);

% Run batch LSE

y = (l - r)*0.1;
Z = [v; y];
Z= Z(1:end, 1:end);
leftside  = H.'/W * H;
rightside = H.'/W * Z;
x_posterior= leftside \ rightside;
plot(x_posterior-x_true)
