function value = linear_cka(X,Y)
% Linear centered-kernel alignment between sample-aligned representations.

assert(size(X,1)==size(Y,1),'CKA requires the same samples in both inputs.');
X = double(X)-mean(double(X),1);
Y = double(Y)-mean(double(Y),1);
cross = X'*Y;
denominator = norm(X'*X,'fro')*norm(Y'*Y,'fro');
value = norm(cross,'fro')^2/max(denominator,eps);
end
