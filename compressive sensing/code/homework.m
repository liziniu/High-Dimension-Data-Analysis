% tveq_phantom_example.m
%
% Phantom reconstruction from samples on 22 radial lines in the Fourier
% plane.
%
% Written by: Justin Romberg, Caltech
% Email: jrom@acm.caltech.edu
% Created: October 2005
%

clear;clc;
path(path, './Optimization');
path(path, './Measurements');
path(path, './Data');


% Phantom 
n = 256;
N = n*n;
X = phantom(n);
x = X(:);
% l-p norm
epsilon = 10;
factor = 10;
p = 0.8;

% number of radial lines in the Fourier domain
L = 15;

% Fourier samples we are given
[M,Mh,mh,mhi] = LineMask(L,n);
OMEGA = mhi;
A = @(z) A_fhp(z, OMEGA);
At = @(z) At_fhp(z, OMEGA, n);

% measurements
y = A(x);

% min l2 reconstruction (backprojection)
xbp = At(y);
Xbp = reshape(xbp,n,n);

% recovery
tic
[gx, gy] = gradient(X);
tvI = sum(sum(sqrt(gx.^2 + gy.^2)));
fprintf('Original TV = %8.3f\n', tvI);

un = Xbp;
i = 0;
step = 0.01;
dis = 1000;
while true
   i = i + 1;
   [gx, gy] = gradient(un);
   p_norm = sum(sum(power(sqrt(gx.^2 + gy.^2 ), p)));
   
   coef = power(sqrt(gx.^2 + gy.^2 + epsilon^2), p-2); 
   %coef_x = power(sqrt(gx.^2 + epsilon^2), p-2);
   %coef_y = power(sqrt(gy.^2 + epsilon^2), p-2);
   tvU_x = coef .* gx;
   tvU_y = coef .* gy;
   
   dn = -divergence(tvU_x, tvU_y);
   j = 0;
   if mod(i, 100) == 0
       for t = 1e-3: -1e-5 : 1e-4
           j = j + 1;
           u_next = un - t*dn;
           y_next = A(u_next(:));
           if abs(norm(y_next) - norm(y)) < sqrt(epsilon)/100
               break;
           end
       end
       step = t;
   end
   un = un - step*dn;
   [gx, gy] = gradient(un);
   p_norm_next = sum(sum(power(sqrt(gx.^2 + gy.^2 ), p)));
   tol = max(max(un-X));
   if max(max(un-X)) < dis
       dis = tol;
       imwrite(un,['figure/best.jpg']);
   end
   if mod(i, 100) == 0
        fprintf('i=%i, j=%i, rate=%.3f \n', i, j, p_norm_next/p_norm);
   end
   if p_norm_next < p_norm * 0.95 || mod(i, 1000) == 0
        fprintf('shrink \n');
        epsilon = epsilon / factor;
        imwrite(un,['figure/', int2str(i), '.jpg']);
   end
   if i > 10000
       break;
   end
end
toc
Xtv = un;
figure();
subplot(1, 3, 1);
imshow(X);
subplot(1, 3, 2);
imshow(Xbp);
subplot(1, 3, 3);
imshow(Xtv);