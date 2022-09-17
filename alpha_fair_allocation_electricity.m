clear all
clc
close all


%% 
%{
dataset path details
%}

Data_Original_temp_1 = csvread('sweden data electricity consumption path');
calender_1=csvread('sweden data calender path',1,0);
[no_hours_1, no_customer_1]=size(Data_Original_temp_1);
dataset_name_1 = 'Sweden';

Data_Original_temp_2 = csvread('australia data electricity consumption path',1,0);
calender_2=csvread('australia data calender path',1,0);
[no_hours_2, no_customer_2]=size(Data_Original_temp_2);
dataset_name_2 = 'Australia';

Data_Original_temp_3 = csvread('ireland data electricity consumption path',1,0);
[no_hours_3, no_customer_3]=size(Data_Original_temp_3);
calender_3=csvread('ireland data calender path',1,0);
dataset_name_3 = 'Ireland';

%% aggregating laods

for i=1:no_customer_1
    
B=sum(reshape(Data_Original_temp_1(:,i),24,no_hours_1/24));

data_1(:,i)=B';
end

for i=1:no_customer_2
    
B=sum(reshape(Data_Original_temp_2(:,i),24,no_hours_2/24));

data_2(:,i)=B';
end

for i=1:no_customer_3
    
B=sum(reshape(Data_Original_temp_3(:,i),24,no_hours_3/24));

data_3(:,i)=B';
end

[no_hours_1, no_customer_1]=size(data_1);
[no_hours_2, no_customer_2]=size(data_2);
[no_hours_3, no_customer_3]=size(data_3);

%% clustering customers

no_clusters_1=582;
no_clusters_3=709;
cluster_1 = kmeans(data_1',no_clusters_1,'MaxIter',2000,'EmptyAction','drop');
cluster_3 = kmeans(data_3',no_clusters_3,'MaxIter',2000,'EmptyAction','drop');

clustered_data_1=cell(1,no_clusters_1);

for i=1:length(cluster_1)
    disp(cluster_1(i,1))
  clustered_data_1{1,cluster_1(i,1)}=  [clustered_data_1{1,cluster_1(i,1)} data_1(:,i)]; 
end
    
clustered_data_3=cell(1,no_clusters_3);

for i=1:length(cluster_3)
  clustered_data_3{1,cluster_3(i,1)}=  [clustered_data_3{1,cluster_3(i,1)} data_3(:,i)]; 
end
 
%% weight selection

w_1=zeros(1,no_clusters_1);

for i=1:no_clusters_1
   w_1(1,i)=sum(cluster_1 == i);
end
w_1=w_1/sum(w_1);

w_3=zeros(1,no_clusters_3);

for i=1:no_clusters_3
   w_3(1,i)=sum(cluster_3 == i);
end
w_3=w_3/sum(w_3);

%% upper and lower bound

day_1=1; %sweden_day
day_2=1; %australia day
day_3=1; %ireland day

for i=1:no_clusters_1
U_1(i,1)=sum(clustered_data_1{1,i}(day_1,:));
end

for i=1:no_clusters_3
U_3(i,1)=sum(clustered_data_3{1,i}(day_3,:));
end





%% sweden with

a=[0 0.5 2 3 5 10 20 50 100 200 500 1000 5000 10000];  %alpha=[0-inf]
n=no_clusters_1;
b=ones(1,n);
U=U_1;
L=zeros(n,1);
s=[0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; % supply
x_1=cell(1,length(a));

for i=1:length(s)
    for j=1:length(a)
        disp('j=')
        disp(j)
        disp('i=')
        disp(i)
S=s(1,i)*sum(U);
alpha=a(1,j);
m=w_1/(1-alpha); %weights + denominator

 cvx_begin
 variable x(n);
 maximize( m*(pow_p(x,1-alpha)) );
 subject to
   x <= U;
   x >= L;
  b*x == S;
 cvx_end
 
 x_1{1,j}(i,:)=x;   %sweden
 
    end
end

%% australian dataset
 a=[0 0.5 2 3 5 10 20 50 100 200 500 1000 5000 10000];  %alpha=[0-inf]
n=no_customer_2;
b=ones(1,n);
U=data_2(day_2,:)';
L=zeros(n,1);
s=[0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; % supplya=[0 0.5 2 3 5 10 20 50 100 200 500 1000 5000 10000];  %alpha=[0-inf]
x_2=cell(1,length(a));

for i=1:length(s)
    for j=1:length(a)
        disp('j=')
        disp(j)
        disp('i=')
        disp(i)
S=s(1,i)*sum(U);
alpha=a(1,j);
m=ones(1,n)/(1-alpha); %weights + denominator

 cvx_begin
 variable x(n);
 maximize( m*(pow_p(x,1-alpha)) );
 subject to
   x <= U;
   x >= L;
  b*x == S;
 cvx_end
 
 x_2{1,j}(i,:)=x;   %australia
 
    end
end
 

%% ireland dataset

 a=[0 0.5 2 3 5 10 20 50 100 200 500 1000 5000 10000];  %alpha=[0-inf]
n=no_clusters_3;
b=ones(1,n);
U=U_3;
L=zeros(n,1);
s=[0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; % supply
x_3=cell(1,length(a));

for i=1:length(s)
    for j=1:length(a)
        disp('j=')
        disp(j)
        disp('i=')
        disp(i)
S=s(1,i)*sum(U);
alpha=a(1,j);
m=w_3/(1-alpha); %weights + denominator

 cvx_begin
 variable x(n);
 maximize( m*(pow_p(x,1-alpha)) );
 subject to
   x <= U;
   x >= L;
  b*x == S;
 cvx_end
 
 x_3{1,j}(i,:)=x;   %ireland
 
    end
end
 
%% proportional fair sweden

n=no_clusters_1;
m=w_1; %weights
b=ones(1,n);
L=zeros(n,1);  % lower bounds
U=U_1; % upper bounds
s=[0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; % supply
for i=1:length(s)
S=s(1,i)*sum(U); % available supply

cvx_begin
 variable x(n);
 maximize( m*log(x) );
 subject to
   x <= U;
   x >= L;
   b*x == S;
 cvx_end

x_proportional_1(i,:)=x;
end

%% proportional fair australia

n=no_customer_2;
m=ones(1,n); %weights
b=ones(1,n);
L=zeros(n,1);  % lower bounds
U=data_2(day_2,:)'; % upper bounds
s=[0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; % supply
for i=1:length(s)
S=s(1,i)*sum(U); % available supply

cvx_begin
 variable x(n);
 maximize( m*log(x) );
 subject to
   x <= U;
   x >= L;
   b*x == S;
 cvx_end

x_proportional_2(i,:)=x;
end

%% proportional fair ireland

n=no_clusters_3;
m=w_3; %weights
b=ones(1,n);
L=zeros(n,1);  % lower bounds
U=U_3; % upper bounds
s=[0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; % supply
for i=1:length(s)
S=s(1,i)*sum(U); % available supply

cvx_begin
 variable x(n);
 maximize( m*log(x) );
 subject to
   x <= U;
   x >= L;
   b*x == S;
 cvx_end

x_proportional_3(i,:)=x;
end


%% proportional fair with seperate variables

% L=zeros(10,1);
% a = 5;
% b = 30;
% U = (b-a).*rand(10,1) + a;
% S=0.8*(sum(U));
% 
% cvx_begin
%  variables x1 x2 x3 x4 x5 x6 x7 x8 x9 x10;
%  maximize( log(x1)+log(x2)+log(x3)+log(x4)+log(x5)+log(x6)+log(x7)+log(x8)+log(x9)+log(x10) );
%  subject to
%    [x1;x2;x3;x4;x5;x6;x7;x8;x9;x10] <= U;
%    [x1;x2;x3;x4;x5;x6;x7;x8;x9;x10] >= L;
%    x1+x2+x3+x4+x5+x6+x7+x8+x9+x10 <= S;
%  cvx_end
% x=[x1;x2;x3;x4;x5;x6;x7;x8;x9;x10];
% x_1=[x U];
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%
% n=10;
% m=ones(1,10);
% b=ones(1,10);
% 
% cvx_begin
%  variable x(n);
%  maximize( m*log(x) );
%  subject to
%    x <= U;
%    x >= L;
%    b*x <= S;
%  cvx_end
% 
% x_1=[U x];

% %% proportional fair with linear programming sweden
% 
% X=[1:10000];
% Y1= log(X);
% P1 = polyfit(X(1:10),Y1(1:10),1);
% P2 = polyfit(X(11:20),Y1(11:20),1);  
% P3 = polyfit(X(21:50),Y1(21:50),1);    
% P4 = polyfit(X(51:100),Y1(51:100),1);
% P5= polyfit(X(101:1000),Y1(101:1000),1);
% P6= polyfit(X(1001:10000),Y1(1001:10000),1);
% 
% scatter(X,Y1,25,'b','*');
% 
% hold on;
% Y1_prime=P1(1)*X(1:10) + P1(2);
% Y2_prime=P2(1)*X(11:20) + P2(2);  
% Y3_prime=P3(1)*X(21:50) + P3(2);    
% Y4_prime = P4(1)*X(51:100)+P4(2);
% Y5_prime = P5(1)*X(101:1000)+P5(2);
% Y6_prime = P6(1)*X(1001:10000)+P6(2);
% plot(X(1:10),Y1_prime,'r+')
% plot(X(11:20),Y2_prime,'g+')  
% plot(X(21:50),Y3_prime,'r+')    
% plot(X(51:100),Y4_prime,'g+')
% plot(X(101:1000),Y5_prime,'r+')
% plot(X(1001:10000),Y6_prime,'g+')
% hold off
% 
% Error = sum(Y1 - [Y1_prime Y2_prime Y3_prime Y4_prime Y5_prime Y6_prime]);
% 
% n=no_clusters_1;
% b=ones(1,n);
% L=zeros(n,1); %lower bound
% U=U_1; % upper bounds
% S=0.8*(sum(U)); % available supply
% 
% 
% A1=diag(b'*P1(1,1));
% B1=(b'*P1(1,2));
% 
% A2=diag(b'*P2(1,1));
% B2=(b'*P2(1,2));
% 
% A3=diag(b'*P3(1,1));
% B3=(b'*P3(1,2));
% 
% A4=diag(b'*P4(1,1));
% B4=(b'*P4(1,2));
% 
% A5=diag(b'*P5(1,1));
% B5=(b'*P5(1,2));
% 
% A6=diag(b'*P6(1,1));
% B6=(b'*P6(1,2));
% 
% cvx_begin
%  variable f(n);
%  variable x(n);
%  maximize( b*f );
%  subject to
%    f-B1 <= A1*x;
%    f-B2 <= A2*x;
%    f-B3 <= A3*x;
%    f-B4 <= A4*x;
%    f-B5 <= A5*x;
%    f-B6 <= A6*x;
%    x <= U;
%    x >= L;
%   b*x == S;
%  cvx_end
%  
%  
%  
%% price bracket two brackets

p1=10;
p2=20;
c=[10 20 30 40 50 60 70 80 90]';
b1= prctile(U_1,c);
b2= prctile(data_2(day_2,:)',c);
b3= prctile(U_3,c);

price_1_prop=0;
price_2_prop=0;
price_3_prop=0;

for i=1:length(s)
        
       for k=1:length(c)
           
           p=0;
           for L=1:length(x_proportional_1(i,:))
              
              if x_proportional_1(i,L) >= b1(k,1)
                  p=p+ b1(k,1)*p1+ p2*(x_proportional_1(i,L)-b1(k,1));
              end
              
              if x_proportional_1(i,L) < b1(k,1)
              
                  p=p+ p1*x_proportional_1(i,L);
              end
           end
           price_1_prop(i,k)=p;
       end
end

for i=1:length(s)
       
       for k=1:length(c)
           
           p=0;
           for L=1:length(x_proportional_2(i,:))
            
              if x_proportional_2(i,L) >= b2(k,1)
                  p=p+ b2(k,1)*p1+ p2*(x_proportional_2(i,L)-b2(k,1));
              end
              
              if x_proportional_2(i,L) < b2(k,1)
              
                  p=p+ p1*x_proportional_2(i,L);
              end
           end
           price_2_prop(i,k)=p;
       end
end
    

for i=1:length(s)
        
       for k=1:length(c)
           
           p=0;
           for L=1:length(x_proportional_3(i,:))
              
              if x_proportional_3(i,L) >= b1(k,1)
                  p=p+ b1(k,1)*p1+ p2*(x_proportional_3(i,L)-b1(k,1));
              end
              
              if x_proportional_3(i,L) < b1(k,1)
              
                  p=p+ p1*x_proportional_3(i,L);
              end
           end
           price_3_prop(i,k)=p;
       end
end

price_1_prop=floor(price_1_prop);
price_2_prop=floor(price_2_prop);
price_3_prop=floor(price_3_prop);

%%   

price_1=cell(1,length(a));
price_2=cell(1,length(a));
price_3=cell(1,length(a));

for j=1:length(a)
    
    for i=1:length(s)
        
       for k=1:length(c)
           
           p=0;
           for L=1:length(x_1{1,j}(i,:))
              
              if x_1{1,j}(i,L) >= b1(k,1)
                  p=p+ b1(k,1)*p1+ p2*(x_1{1,j}(i,L)-b1(k,1));
              end
              
              if x_1{1,j}(i,L) < b1(k,1)
              
                  p=p+ p1*x_1{1,j}(i,L);
              end
           end
           price_1{1,j}(i,k)=p;
       end
    end
end
 

for j=1:length(a)
    
    for i=1:length(s)
       
       for k=1:length(c)
           
           p=0;
           for L=1:length(x_2{1,j}(i,:))
            
              if x_2{1,j}(i,L) >= b2(k,1)
                  p=p+ b2(k,1)*p1+ p2*(x_2{1,j}(i,L)-b2(k,1));
              end
              
              if x_2{1,j}(i,L) < b2(k,1)
              
                  p=p+ p1*x_2{1,j}(i,L);
              end
           end
           price_2{1,j}(i,k)=p;
       end
    end
end


for j=1:length(a)
    
    for i=1:length(s)
       
       for k=1:length(c)
           
           p=0;
           for L=1:length(x_3{1,j}(i,:))
            
              if x_3{1,j}(i,L) >= b3(k,1)
                  p=p+ b3(k,1)*p1+ p2*(x_3{1,j}(i,L)-b3(k,1));
              end
              
              if x_3{1,j}(i,L) < b3(k,1)
              
                  p=p+ p1*x_3{1,j}(i,L);
              end
           end
           price_3{1,j}(i,k)=p;
       end
    end
end

%%

f1=[1 2 3 5 6 9 12 14];
f2=[9 7 4 1];
f3=[1 5 9];

table_1=0;
table_2=0;
table_3=0;
for j=1:length(f1)
    d=1;
  for k=1:length(f3)
      for i=1:length(f2)
    table_1(j,d)= price_1{1,f1(j)}(f2(i),f3(k));
    d=d+1;
      end
  end
end

for j=1:length(f1)
    d=1;
  for k=1:length(f3)
      for i=1:length(f2)
    table_2(j,d)= price_2{1,f1(j)}(f2(i),f3(k));
    d=d+1;
      end
  end
end

for j=1:length(f1)
    d=1;
  for k=1:length(f3)
      for i=1:length(f2)
    table_3(j,d)= price_3{1,f1(j)}(f2(i),f3(k));
    d=d+1;
      end
  end
end
table_1=floor(table_1);
table_2=floor(table_2);
table_3=floor(table_3);
% i=s=[0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]; % supplya=[0 0.5 2 3 5 10 20 50 100 200 500 1000 5000 10000];  %j=alpha=[0-inf]
 %% plots
 
 figure()
 subplot(3,1,1)
 bar(w_1)
 xlabel('Cluster')
 ylabel('Weight')
 title('Sweden')
 
 subplot(3,1,2)
 bar(ones(1,no_customer_2))
 ylim([0 2])
 xlabel('Customers')
 ylabel('Weight')
 title('Australia')
 
 subplot(3,1,3)
 bar(w_3)
 xlabel('Cluster')
 ylabel('Weight')
 title('Irish')
 
 %% effect of S on allocation alpha=0

 y(:,1)=data_2(day_2,1:5)';
 y(:,2)=x_2{1,1}(1,1:5);
 y(:,3)=x_2{1,1}(5,1:5);
 y(:,4)=x_2{1,1}(9,1:5);
 bar(y)
legend({'Demand','90% Supply','50% Supply','10% Supply'},'Location','northwest')
xlabel('Customers')
 ylabel('Allocation')
 
 %% effect of S on allocation alpha=2
clear y
 y(:,1)=data_2(day_2,1:5)';
 y(:,2)=x_2{1,3}(1,1:5);
 y(:,3)=x_2{1,3}(5,1:5);
 y(:,4)=x_2{1,3}(9,1:5);
 bar(y)
legend({'Demand','90% Supply','50% Supply','10% Supply'},'Location','northwest')
xlabel('Customers')
 ylabel('Allocation')
 
 %% effect of S on allocation alpha=10000
 clear y
 y(:,1)=data_2(day_2,1:5)';
 y(:,2)=x_2{1,14}(1,1:5);
 y(:,3)=x_2{1,14}(5,1:5);
 y(:,4)=x_2{1,14}(9,1:5);
 bar(y)
legend({'Demand','90% Supply','50% Supply','10% Supply'},'Location','northwest')
xlabel('Customers')
ylabel('Allocation')
 
 %% effect of S on allocation alpha=1
 clear y
 y(:,1)=data_2(day_2,1:5)';
 y(:,2)=x_proportional_2(1,1:5);
 y(:,3)=x_proportional_2(5,1:5);
 y(:,4)=x_proportional_2(9,1:5);
 bar(y)
legend({'Demand','90% Supply','50% Supply','10% Supply'},'Location','northwest')
xlabel('Customers')
ylabel('Allocation')
 
 %% effect of alpha on allocation s=90%
clear y
 y(:,1)=data_2(day_2,:)';
 y(:,2)=x_2{1,1}(1,:);
 y(:,3)=x_proportional_2(1,:);
 y(:,4)=x_2{1,3}(1,:);
 y(:,5)=x_2{1,14}(1,:);
 plot(y)
legend({'Demand','alpha=0','alpha=1','alpha=2','alpha=10000'},'Location','northwest')
xlabel('Customers')
ylabel('Allocation')
 
 
 %% effect of alpha on allocation s=50%
 clear y
 y(:,1)=data_2(day_2,1:5)';
 y(:,2)=x_2{1,1}(5,1:5);
 y(:,3)=x_proportional_2(5,1:5);
 y(:,4)=x_2{1,3}(5,1:5);
 y(:,5)=x_2{1,14}(5,1:5);
 bar(y)
legend({'Demand','alpha=0','alpha=1','alpha=2','alpha=10000'},'Location','northwest')
xlabel('Customers')
ylabel('Allocation')
 
 %% effect of alpha on allocation s=10%
 clear y
 y(:,1)=data_2(day_2,1:5)';
 y(:,2)=x_2{1,1}(9,1:5);
 y(:,3)=x_proportional_2(9,1:5);
 y(:,4)=x_2{1,3}(9,1:5);
 y(:,5)=x_2{1,14}(9,1:5);
 bar(y)
legend({'Demand','alpha=0','alpha=1','alpha=2','alpha=10000'},'Location','northwest')
xlabel('Customers')
ylabel('Allocation')
 
 %%
 
histogram(x_1{1,1}(1,:),20)
hold on
histogram(x_1{1,1}(5,:),20)
%hold on
%histogram(x_1{1,1}(,:),20)
 
 
 
 
 
 
 
 
 
 