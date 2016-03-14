function out = obj_sparse(ratio,m,n)

N = m*n;
out = zeros(N,1);
S = N*ratio;
rseq = randperm(N);
out(rseq(1:S)) = rand([1,S]);
out = reshape(out,m,n);
