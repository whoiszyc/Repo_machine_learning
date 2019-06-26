close all;
clc;
clear all;
warning off;
% This is just to simulate the system
% A=[-0.7 1.8;2.5 0.97];
% B=[2;2.6];
% C=0.5*[1 1]; 
% F=[0.97];
A=[-1 2;2.2 1.7];
B=[2;1.6];
C=[1 2];
F=[-1];
L=[0 0 0];
R=1;Q=6;
gamma=0.8;
Q1=[C'*Q*C -C'*Q;-Q*C Q];
G=[Q1 [0 ; 0;0];[0 0 0] R];
T=[A [0;0];[0 0] F];
B1=[B;0];
x=[1;-1];
r=[1];
P1=dare(sqrt(gamma)*T,sqrt(gamma)*B1,Q1,R);
H1=[1 0 0 1;0 1 0 1;0 1 1 0;-1.6713 1.4279 0 1];
% H1=[1 0 0 1;0 1 0 1;0 1 1 0;1 0 0 1];

%%%%%%%%%%%   offline solution

H1yy=H1(4,4);H1yx=H1(4,1:3);
L2=-inv(H1yy)*H1yx;
L1=-inv(H1yy)*H1yx;
for i=1:70
    H1=G+gamma*[T B1;L1*T L1*B1]'*H1*[T B1;L1*T L1*B1];
    H1yy=H1(4,4);H1yx=H1(4,1:3);
    L1=-inv(H1yy)*H1yx;
end
% H2=[Q1+gamma*T'*P1*T gamma*T'*P1*B1;gamma*(B1)'*P1*T R+gamma*(B1)'*P1*B1]; 
% 
% X(:,1)=[5;-5;5];
% for j=1:200
%     u=L1*X(:,j);
% X(:,j+1)=T*X(:,j)+B1*u
% y(j)=C*X(1:2,j);
% end
% 
% t=1:200;
% plot(t,y,'r',t,X(3,1:200),'b')
% 
% 
% 

X=[5;-5;5];

x=[5;-5];
r=[5];
% H=[1 0 0 1;0 1 0 1;0 1 1 0;-1.6713 1.4279 0 1];

% H=[330.7296 -234.4091   -0.2078 -136.8702
%  -234.3657  227.9646   -9.8064  169.8777
%   -29.1459   13.8630    2.3258    5.9984
%  -136.8131  169.8547   -9.8608  152.2920];

%%%%%%%%%%%%% online solution

H=[110.0070 -75.76  32.5366 -80.7880
 -75.7695  550.7413  -45.3487  530.1231
   32.5366  -45.3487   2.4461  -5.4254
 -80.7880  530.1231  -5.4254  566.0];

Hyy=H(4,4);Hyx=H(4,1:3);L=-inv(Hyy)*Hyx;
zbar(1:10,1:21)=0;d_target(1:21,1)=0;
eig(H);
eig(A+B*L(1:2));
Hyy=H(4,4);Hyx=H(4,1:3);L=-inv(Hyy)*Hyx;
zbar(1:10,1:21)=0;d_target(1:21,1)=0;
kk=0;
for i=1:1000
    i
    X(:,i)=[x(:,i);r(:,i)];
%    
%     r(:,i+1)=F*r(:,i);
%     u=L*X(:,i);
%     Z(:,i)=[X(:,i);u];
    a1=0;
    a2=0.97;
    
    if i>=300
        a1=0;
        a2=0;
    end
    
    
    r(:,i+1)=F*r(:,i)+a1*(0.1*sin(2*i)^3*cos(9*i)+0.37*sin(1.1*i)^2*cos(4.00*i)+0.3*sin(2.2*i)^4*cos(7.*i)+0.3*sin(10.3*i)^2+0.7*sin(3*i)^2*cos(4*i)+0.3*sin(3*i)*cos(1.2*i)^2+0.4*sin(1.12*i)^3+0.5*cos(2.4*i)*sin(8*i)^2+0.3*sin(1*i)*cos(0.8*i)^2+0.3*sin(4*i)^3+0.4*cos(2*i)*sin(5*i)^4+0.3*sin(5*i)^5);
    u=L*X(:,i)+a2*(0.5*sin(2.0*i)^2*cos(10.1*i)+0.9*sin(1.102*i)^2*cos(4.001*i)+0.3*sin(1.99*i)^2*cos(7*i)+0.3*sin(10.0*i)^3+0.7*sin(3.0*i)^2*cos(4.0*i)+0.3*sin(3.00*i)*1*cos(1.2*i)^2+0.400*sin(1.12*i)^2+0.5*cos(2.4*i)*sin(8*i)^2+0.3*sin(1.000*i)^1*cos(0.799999*i)^2+0.3*sin(4*i)^3+0.4*cos(2*i)*1*sin(5*i)^4+0.3*sin(10.00*i)^3);
    Z(:,i)=[X(:,i);u];
%     
    
    x(:,i+1)=A*x(:,i)+B*u;
    y(i)=C*x(:,i);
    X(:,i+1)=[x(:,i+1);r(:,i+1)];
    d_target(1,1)=d_target(2,1);
    d_target(2,1)=d_target(3,1);
    d_target(3,1)=d_target(4,1);
    d_target(4,1)=d_target(5,1);
    d_target(5,1)=d_target(6,1);
    d_target(6,1)=d_target(7,1);
    d_target(7,1)=d_target(8,1);
    d_target(8,1)=d_target(9,1);
    d_target(9,1)=d_target(10,1);
    d_target(10,1)=d_target(11,1);
    d_target(11,1)=d_target(12,1);
    d_target(12,1)=d_target(13,1);
    d_target(13,1)=d_target(14,1);
    d_target(14,1)=d_target(15,1);
    d_target(15,1)=d_target(16,1);
    d_target(16,1)=d_target(17,1);
    d_target(17,1)=d_target(18,1);
    d_target(18,1)=d_target(19,1);
    d_target(19,1)=d_target(20,1);
    d_target(20,1)=d_target(21,1);
    d_target(21,1)=[X(:,i); u]'*G*[X(:,i); u]+gamma*[X(:,i+1);L*X(:,i+1)]'*H*[X(:,i+1);L*X(:,i+1)];    % the more you include the better
    zbar(:,1)=zbar(:,2);
    zbar(:,2)=zbar(:,3);
    zbar(:,3)=zbar(:,4);
    zbar(:,4)=zbar(:,5);
    zbar(:,5)=zbar(:,6);
    zbar(:,6)=zbar(:,7);
    zbar(:,7)=zbar(:,8);
    zbar(:,8)=zbar(:,9);
    zbar(:,9)=zbar(:,10);
    zbar(:,10)=zbar(:,11);
    zbar(:,11)=zbar(:,12);
    zbar(:,12)=zbar(:,13);
    zbar(:,13)=zbar(:,14);
    zbar(:,14)=zbar(:,15);
    zbar(:,15)=zbar(:,16);
    zbar(:,16)=zbar(:,17);
    zbar(:,17)=zbar(:,18);
    zbar(:,18)=zbar(:,19);
    zbar(:,19)=zbar(:,20);
    zbar(:,20)=zbar(:,21);
    zbar(:,21)=[X(1,i)^2;X(1,i)*X(2,i);X(1,i)*X(3,i) ;X(1,i)*u;X(2,i)^2;X(2,i)*X(3,i);X(2,i)*u;X(3,i)^2;X(3,i)*u;u^2];
   
    if mod(i,21)==0
        eL=abs(L-L1);
        NeH=abs(H1-H);
        kk=kk+1;
        if i<=300
        m=zbar*zbar';q=zbar*d_target;
        rank(m)
        vH=inv(m)*q;
        H=[vH(1,1) vH(2,1)/2 vH(3,1)/2 vH(4,1)/2 ; vH(2,1)/2 vH(5,1) vH(6,1)/2 vH(7,1)/2;vH(3,1)/2 vH(6,1)/2 vH(8,1) vH(9,1)/2;vH(4,1)/2 vH(7,1)/2 vH(9,1)/2 vH(10,1)]        
        Hyy=H(4,4);Hyx=H(4,1:3);
        L=-inv(Hyy)*Hyx; 
        end
      
    d(kk)=norm(eL)
   d1(kk)=norm(H1-H)
    end
    
end
eH=abs(H1-H)
o=norm(eH)
H
H1
t=1:1000;
plot(t,y,'r',t,r(1:1000),'g')
figure(2)
t1=1:kk;
plot(t1,d(1:kk),'o')
hold on
plot(t1,d(1:kk),'r')
figure(3)
t2=1:kk;
plot(t2,d1(1:kk),'o')
hold on
plot(t2,d1(1:kk),'r')
