function odestart
clear all;close all;clc;
global P;
global T
global B1
global R
global Q1
global u
global gamma

%system matrices
A=[0.5 1.5;2 -2];
E=eig(A)
B=[4 1]';

C=[1 0];
P=rand(3,3);
R=1;
F=[0];
Q=5;
B1=[B;0];
T=[A [0;0];[0 0] F];
C1=[C -1];
x0=[-1,1,5, 0];
Q1=C1'*Q*C1;
Q11=C'*Q*C;
PP=care(T-0.5*0.1*eye(3),B1,Q1,1)
LL=-inv(R)*B1'*PP;
kk=0;
uu=[]; % saving the control signal for plot

%parameters of the critic rearanged as it would be returned by the least
%squares
WW=[P(1,1); 2*P(1,2); 2*P(1,3); P(2,2); 2*P(2,3); P(3,3)];
WWP=[WW; 0];

Fsamples=100; %length of the simulation in samples 
T1=.05; % sample time
gamma=0.1;
%parameters for the batch least squares
j=0;
Xpi=[];

upd=[]; % stores information relative to updates of the critic parameters
ch=0;

for k=1:Fsamples,
    j=j+1;
    X(j,:)=[x0(1)^2 x0(1)*x0(2) x0(1)*x0(3) x0(2)^2 x0(2)*x0(3) x0(3)^2];
    before_cost=[x0(1) x0(2) x0(3)]*P*[x0(1) x0(2) x0(3)]';
    tspan=[0 T1];
    [t,x]= ode45(@odefile,tspan,x0);
    x1=x(length(x),1);
    x2=x(length(x),2);
    x3=x(length(x),3);
    after_cost=x(length(x),4)+exp(-gamma*T1)*[x1 x2 x3]*P*[x1;x2;x3];
    Xpi(j,:)=X(j,:)-exp(-gamma*T1)*[x1^2 x1*x2 x1*x3 x2^2 x2*x3 x3^2];
%     figure(1); 
    plot(t+T1*(k-1),x(:,1),'-b',t+T1*(k-1),x(:,3),'-g','LineWidth',2);
    hold on
    Y(j,:)=x(length(x),4);
    x0=[x(length(t),1) x(length(t),2) x(length(t),3) 0];
%     plot(t(length(t))+T1*(k-1),x0(1:3),'o');
    
    uu=[uu u];
    
    if (abs(after_cost-before_cost)>0.00001)&&(ch==0)&&(mod(j,6)~=1),
        j=0;
        ch=ch+1;
    else
        if abs(after_cost-before_cost)>0.00001,
        ch=ch+1;
        end
    end
    
    
    % the batch least squares is made on 6 values
    if mod(j,6)==0,
        kk=kk+1;
        if (abs(after_cost-before_cost)>0.00001)&&(ch==6),
        weights=Xpi\Y; %calculation of the weights
        upd=[upd 1];
        else 
        %there is no reason to update
        upd=[upd 0];       
        end
        WWP=[WWP [weights; k*T1]];
        WW=[WW weights];
         %calculating the matrix P
        P=[weights(1) weights(2)/2 weights(3)/2 ; weights(2)/2 weights(4) weights(5)/2; weights(3)/2 weights(5)/2 weights(6)];
        L=-inv(R)*B1'*P
        j=0;
        ch=0;
        d1(kk)=norm(P-PP);
        d11(kk)=norm(L-LL);
%         E=[E eig(T-B1*B1'*P)];
    end
end




figure(1); title('System output versus reference trajectory'); xlabel('Time(s)');

%printing for comparison
sol=care(T-0.5*0.1,B1,Q1,1)
P=[weights(1) weights(2)/2 weights(3)/2 ; weights(2)/2 weights(4) weights(5)/2; weights(3)/2 weights(5)/2 weights(6)]

WW=[WW [sol(1,1); 2*sol(1,2); 2*sol(1,3); sol(2,2); 2*sol(2,3); sol(3,3)]];

figure; plot([0:T1:T1*(length(uu)-1)],uu,'LineWidth',2);
title('Control signal'); xlabel('Time(s)');

%plotting the poles of the closed loop system
% figure;
% for jj=1:Fsamples/6,
%     plot(6*T*(jj-1),real(E(:,jj)),'.');
%     hold on;
% end
% title('Poles of the closed loop system'); xlabel('Time(s)')

% figure;
% hold on;
% plot(WWP(7,:)',WWP(1:6,:)','.-');
% title('P matrix parameters'); xlabel('Time(s)')

% figure;
% WWP=[WWP [[sol(1,1); 2*sol(1,2); 2*sol(1,3); sol(2,2); 2*sol(2,3); sol(3,3)]; T*(Fsamples-1)]];
% xy=size(WWP);
% plot(WWP(7,1:(xy(2)-1)),WWP(2:2:6,1:(xy(2)-1)),'.');
% hold on;
% plot(WWP(7,xy(2)),WWP(2:2:6,xy(2)),'*');
% title('P matrix parameters'); xlabel('Time(s)'); axis([0 3 -0.2 2.5]);

% figure;
% plot(upd,'*'); title('P parameters updates'); xlabel('Iteration number');axis([0 10 -0.1 1.1]);
% % 
% 
% tspan=[0 50];
% [t,x]=ode45(@odefile,tspan,x0);
% y=C*x(:,1:2)';
% plot(t,y,'r',t,x(:,3),'b') 
   
g=1:kk;
figure;
plot(g,d1,'LineWidth',2)
title('Convergence of the P matrix to the optimal P^{*} matrix ','LineWidth',24);; xlabel('Number of iterations','LineWidth',24);
hold on
plot(g,d1,'o','LineWidth',2)

figure;
plot(g,d11,'LineWidth',2)
title('Convergence of the control gain K_{1} to the optimal control gain K_{1}^{*} ','LineWidth',24); xlabel('Number of iterations','LineWidth',24);
hold on
plot(g,d11,'o','LineWidth',2)

    
function xdot=odefile(t,x)
global P;
global T
global B1
global R
global Q1
global u
global gamma
x1=x(1);
x2=x(2);
x3=x(3);

  %calculating the control signal
u=-inv(R)*B1'*P*[x1;x2;x3];

  %updating the derivative of the state=[x(1:2) V]
xdot=[T*[x1;x2;x3]+B1*u
      exp(-gamma*t)*[x1;x2;x3]'*Q1*[x1;x2;x3]+u'*R*u];
  
    
