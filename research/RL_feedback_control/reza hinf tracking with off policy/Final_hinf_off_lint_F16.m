
function []=engine_main()
global nn
global T
clc;
x_save=[];
t_save=[];
flag=1; % 1: learning is on. 0: learning is off.
% System matrices used for simulation purpose
A=[-1.01887 0.90506 -0.00215 -1.01887 0.90506 -0.00215
    0.82225 -1.07741 -0.17555 0.82225 -1.07741 -0.17555
    0 0 -1 0 0 -1
    0 0 0 0 0 0
    0 0 0 0 0 0
    0 0 0 0 0 0];

% A=[-1.01887 0.90506 -0.00215 0 0 0
%     0.82225 -1.07741 -0.17555 0 0 0
%     0 0 -1 0 0 0
%     0 0 0 0 0 0
%     0 0 0 0 0 0
%     0 0 0 0 0 0];


B=[0;0;5;0;0;0];
D=[1;0;0;0;0;0];
% C=[eye(3) -eye(3)];
QQ=[20 0 0 0 0 0;0 0 0 0 0 0 ;0 0 0 0 0 0;0 0 0 0 0 0;0 0 0 0 0 0;0 0 0 0 0 0]

[xn,un]=size(B);%size of B. un-column #, xn row #
% Set the weighting matrices for the cost function
gamma=10;
wn=size(D,2);
Q=QQ;
R=1*eye(un);
% Initialize the feedback gain matrix
Kx=[0   0    0    0    0    0];
Kx=[0   0    0    0   0   0];

% Only if A is Hurwitz, K can be set as zero.
% KK=[-0.1159   -0.1293    0.4159    0.1095+0.1159    0.1207+0.1293  -0.2052-0.4159];  % Only if A is Hurwitz, K can be set as zero.
BB = [D,B];
m1 = size(D,2);
m2 = size(B,2);
R1 = [-gamma^2*eye(m1) zeros(m1,m2) ; zeros(m2,m1) eye(m2)]'
Xa = care(A-0.05*eye(xn),BB,QQ,R1)
K0=inv(R)*B'*Xa;
KK=[0   0    0.0005    0    0    0.0005];
% [M,P0]=lqr(A-0.0005*eye(xn),B,C'*QQ*C,R) % Calculate the ideal solution for comparion purpose
% K0=M;
% KK=M;
Kw=zeros(wn,xn);
N=200; %Length of the window, should be at least greater than xn^2
NN=10; %Max iteration times
T=.1; %Duration of time for each integration
x0=[2;1;-2;2;3;1]; %Initial condition
i1=(rand(1,1000)-.5)*100;
i2=(rand(1,1000)-.5)*100;
i3=(rand(1,1000)-.5)*100;
i4=(rand(1,1000)-.5)*100;
i5=(rand(1,1000)-.5)*100;

Dxx=[];XX=[];XU=[]; XW=[];% Data matrices
X=[x0;kron(x0',x0')';kron(x0,zeros(un,1));kron(x0,zeros(wn,1))]';
% Run the simulation and obtain the data matrices \delta_{xx}, I_{xx},

for i=1:N
    nn=i;
% Simulation the system and at the same time collect online info.
[t,X]=ode45(@mysys, [(i-1)*T,i*T],X(end,:));
%Append new data to the data matrices
Dxx=[Dxx;exp(-0.01)*kron(X(end,1:xn),X(end,1:xn))-kron(X(1,1:xn),X(1,1:xn))];
XX=[XX;X(end,xn+1:xn+xn^2)-X(1,xn+1:xn+xn^2)];
XU=[XU;X(end,xn+xn^2+1:xn+xn^2+xn*un)-X(1,xn+xn^2+1:xn+xn^2+xn*un)];
XW=[XW;X(end,xn+xn^2+xn*un+1:end)-X(1,xn+xn^2+xn*un+1:end)];
x_save=[x_save;X];
t_save=[t_save;t];
% if i>1500
%     X(end,4)=3;
% end
% plot(t,X(:,1),'b',t,X(:,4),'k', 'LineWidth',2);
% hold on
% legend('Output','Reference trajectory')
end
Dxx=processing_Dxx(Dxx); % Only the distinct columns left
% K=zeros(un,xn); % Initial stabilizing feedback gain matrix
P_old=zeros(xn);P=eye(xn)*10; % Initialize the previous cost matrix
it=0; % Counter for iterations
p_save=[]; % Track the cost matrices in all the iterations
k_save=[]; % Track the feedback gain matrix in each iterations


% BB = [D,B];
% m1 = size(D,2);
% m2 = size(B,2);
% R1 = [-gamma^2*eye(m1) zeros(m1,m2) ; zeros(m2,m1) eye(m2)]'
% X = care(A-0.05*eye(xn),BB,Q,R1)
% K0=inv(R)*B'*X
% K0w=gamma^-2*D'*X
% [K01,P0]=lqr(A-0.05*eye(xn),B,Q,R) % Calculate the ideal solution for comparion purpose
k_save=[norm(Kx-K0)];
p_save=[100];

while it<20 % Stopping criterion for learning
it=it+1 % Update and display the # of iters
P_old=P; % Update the previous cost matrix
QK=Q+Kx'*R*Kx-gamma^2*Kw'*Kw; % Update the Qk matrix
X2=XX*kron(eye(xn),Kx'); %
X3=XX*kron(eye(xn),Kw'); %
X1=[Dxx,-X2-XU,(X3+XW)]; % Left-hand side of the key equation
Y=-XX*QK(:); % Right-hand side of the key equation
pp=X1\Y; % Solve the equations in the LS sense
P=reshape_p(pp); % Reconstruct the symmetric matrix
p_save=[p_save,norm(P-Xa)]; % Keep track of the cost matrix
BPv=pp(end-((xn*un)+(xn*wn)-1):end);
Kx=R*BPv(1:xn)/2;
Kx=Kx'
Kw=BPv(xn+1:end)/(2*gamma^2);
Kw=Kw'
k_save=[k_save,norm(Kx-K0)]; % Keep track of the control gains
p_save=[k_save,norm(Xa-P)]; % Keep track of the control gains

end


figure(4)
plot([0:length(k_save)-1],k_save,'^',[0:length(k_save)-1],k_save)
% axis([-0.5,5,-.5,2])
legend('||K-K^*||')
xlabel('Number of iterations')

function dX=mysys(t,X)
%global A B xn un i1 i2 K flag
x=X(1:xn);
% if t>=0; % See if learning is stopped
% flag=0;
% end
% if flag==1
u1=zeros(un,1);
u2=zeros(un,1);
u3=zeros(un,1);
u4=zeros(un,1);
u5=zeros(un,1);



for lll=i1
u1=u1+sin(lll*t)/length(i1); % constructing the
% exploration noise
end

for lll1=i2
u2=u2+sin(lll1*t)/length(i2); % constructing the
% exploration noise
end

for lll2=i3
u3=u3+sin(lll2*t)/length(i3); % constructing the
% exploration noise
end

for lll3=i4
u4=u4+sin(lll3*t)/length(i4); % constructing the
% exploration noise
end

for lll4=i5
u5=u5+sin(lll4*t)/length(i5); % constructing the
% exploration noise
end

w=80*u2;
u1=80*u1;
% else
u=-KK*x+u1;
D1=[0;0;0;0.01;0;0];
D2=[0;0;0;0;0.01;0];
D3=[0;0;0;0;0;0.01];

ww1=10*u3;
ww2=10*u4;
ww3=10*u5;


% ww=0;
% end
dx=A*x+B*u+D*w+D1*ww1+D2*ww2+D3*ww3;

dxx=exp(-0.1*(t-(nn-1)*T))*kron(x',x')';
dux=exp(-0.1*(t-(nn-1)*T))*kron(x',u')';
dwx=exp(-0.1*(t-(nn-1)*T))*kron(x',w')';

dX=[dx;dxx;dux;dwx];
end
% This nested function reconstruct the P matrix from its distinct elements
function P=reshape_p(p)
P=zeros(xn);
ij=0;
for i=1:xn
for j=1:i
ij=ij+1;
P(i,j)=p(ij);
P(j,i)=P(i,j);
end
end
end
% The following nested function removes the repeated columns from Dxx
function Dxx=processing_Dxx(Dxx)
ij=[]; ii=[];
for i=1:xn
ii=[ii (i-1)*xn+i];
end
for i=1:xn-1
for j=i+1:xn
ij=[ij (i-1)*xn+j];
end
end
Dxx(:,ii)=Dxx(:,ii)/2;
Dxx(:,ij)=[];
Dxx=Dxx*2;
end
end
