% This is Iterative Shrinkage Thresholding Algorithm (ISTA) for solving 
% LASSO problem. LASSO problem assumes that signal x be sparse, and this
% assumption is not wrong. Most natural siggnal can be represented sparse
% in some domain. For example, natural scenes are sparse in Wavelet
% transform domain or DCT domain. Sometimes the scene itself can be very
% sparse (e.g. stars at night).
%
% ISTA is a first-order method which is gradient-based so it is simple
% and efficient. However, its convergence is slow - O(1/k). A fast ISTA
% (FISTA) is developed for faster convergence, which gives an improved
% complexity, O(1/(k^2)).
%
% Here we will compare the LASSO problem with ISTA and FISTA to CLS problem
% with CG. You will see ISTA and FISTA work well on the sparse signal while
% CLS doesn't. You will see the improved performance of FISTA over ISTA as 
% well.
%
%  Degradation model (sparse model): 
%      y = Hx = HPb = Ab
%    , where A = HP, P:representation matrix (P = I in my example)
%  Deconvolution (LASSO): 
%      min_x ||y-Ab||2 + lambda*||b||1
%
% Author: Seunghwan Yoo (seunghwanyoo2013@u.northwestern.edu)

clear; close all;
addpath(genpath('.'));

param.blur = 1; % 1:Gaussian kernel, 2:User defined
param.obs = 2; % 1:part of peppers.png, 2:random points

%% original image
if param.obs == 1
    x0_whole = im2double(imread('peppers.png'));%'greens.jpg';
    if ndims(x0_whole) > 1
        x0_whole = rgb2gray(x0_whole);
    end
    x_2d = x0_whole(201:220,201:220); % original image (20x20)
elseif param.obs == 2
    sparsity = 0.01;
    %x_2d = obj_sparse(sparsity,20,20);
    load x_2d;
end
x = x_2d(:); % vectorized

%% blur kernel & laplacian kernel
switch (param.blur)
    case 1
        h0_2d = fspecial('gaussian',[11,11],1.2);
    case 2
        h0_2d = [1 1 1; 1 1 1; 1 0 0];%h0 = ones(5,5); % blur kernel
        h0_2d = h0_2d/sum(sum(h0_2d)); % blur kernel
end
c0_2d = [0 0.25 0; 0.25 -1 0.25; 0 0.25 0]; % 2D Laplacian for CLS

%% create operator matrix for lexicographic notation
tic; [h,h_2d] = create_lexicoH(x_2d,h0_2d); toc;
tic; [c,c_2d] = create_lexicoH(x_2d,c0_2d); toc;

%% degradation
fprintf('\n== Degradation\n');
y = h*x;                            % blurred image (vector)
y_2d = reshape(y,size(x_2d));     % blurred image (2D);
% y_2d = imnoise(y_2d_b,'gaussian',0,0.01);  % noisy image (2D);
% y = y_2d(:);                               % noisy image (vector);
figure('Position',[500,500,300,300]),imagesc(x_2d);title('original');caxis([0,1]);
figure('Position',[500,500,300,300]),imagesc(y_2d);title('degraded');caxis([0,1]);


%% non-blind deconvolution (without noise, known y,h, get x)
opt.linesearch = 1; % 1:wolfe, 2:backtracking
opt.rho = 0.5;      % param for backtracking line search
opt.tol = 10^(-6);  % param for stopping criteria
opt.maxiter = 2000; % param for max iteration
opt.lambda = 0.01;  % param for regularizing param
opt.c = c;          % param for CLS, regularizing matrix
opt.fistamode = 2;  % param for FISTA, 1/2
opt.vis = 0;        % param for display, 0:nothing,1:log,2:log+figure
obj.func = @func3;  % func1:LS, func2:CLS w/ I, func3:CLS w/ C
obj.grad = @func3_grad;
obj.hess = @func3_hess;
x0 = y;

%%% ISTA
fprintf('== LASSO with ISTA\n');
[x_ista_i] = opt_ista_lasso(h,y,x0,opt,0);
figure('Position',[500,500,300,300]),imagesc(reshape(x_ista_i,size(x_2d)));title('LASSO - ISTA');caxis([0,1]);
psnr_ista_i = psnr(x_2d,x_ista_i,1);
fprintf('psnr: %.2f\n',psnr_ista_i);

%%% FISTA
fprintf('== LASSO with FISTA\n');
[x_fista_i] = opt_fista_lasso(h,y,x0,opt,0);
figure('Position',[500,500,300,300]),imagesc(reshape(x_fista_i,size(x_2d)));title('LASSO - FISTA');caxis([0,1]);
psnr_ista_i = psnr(x_2d,x_fista_i,1);
fprintf('psnr: %.2f\n',psnr_ista_i);

%%% CLS
opt.lambda = 0.1;    % param for CLS, regularizing param
fprintf('== CLS with CG\n');
[x_cls_i] = opt_cg(obj,x0,opt,y,h);
figure('Position',[500,500,300,300]),imagesc(reshape(x_cls_i,size(x_2d)));title('CLS - CG');caxis([0,1]);
psnr_cls_i = psnr(x_2d,x_cls_i,1);
fprintf('psnr: %.2f\n',psnr_cls_i);
