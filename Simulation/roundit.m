function [y,options] = roundit(x,options)

if nargin < 2 || isempty(options) 
   options.round = 1; options.flip = 0; options.p = 0.5;
end    
if ~isfield(options,'round'), options.round = 1; end
if ~isfield(options,'flip'), options.flip = 0; end
if ~isfield(options,'p'), options.p = 0.5; end

mysign = @(x) sign(x) + (x==0); % mysign(0) = 1;

switch options.round
  
  case 1
    y = abs(x);
    % Built-in ROUND function rounds ties away from zero.
    % Round to nearest integer using round to even to break ties.
    u = round(y - (rem(y,2) == 0.5));
    u(find(u == -1)) = 0; % Special case, negative argument to ROUND.
    y = sign(x).*u; 

  case 2
    % Round towards plus infinity.
    y = ceil(x); 

  case 3
    % Round towards minus infinity.
    y = floor(x); 

  case 4
    % Round towards zero.
    y = (x >= 0 | x == -inf) .* floor(x) + (x < 0 | x == inf) .* ceil(x);

  case {5, 6}

    % Stochastic rounding.
    y = abs(x); 
    frac = y - floor(y);
    k = find(frac ~= 0);
    if isempty(k)
       y = x; 
    else   
      rnd = rand(length(k),1);
      vals = frac(k);  vals = vals(:);

      switch options.round
        case 5 % Round up or down with probability prop. to distance.
               j = (rnd <= vals);
        case 6 % Round up or down with equal probability.       
               j = (rnd <= 0.5);
      end      
      y(k(j)) = ceil(y(k(j)));
      y(k(~j)) = floor(y(k(~j)));
      y = mysign(x).*y; 
   end   
   
  otherwise
    error('Unsupported value of options.round.')  
               
end

if options.flip
    
   temp = rand(size(y));
   k = find(temp <= options.p); % Indices of elements to have a bit flipped.
   if ~isempty(k)
      u = abs(y(k));
      % Random bit flip in significand.
      % b defines which bit (1 to p-1) to flip in each element of y.
      % Using SIZE avoids unwanted implicit expansion.
      b = randi(options.params(1)-1,size(u,1),size(u,2));
      % Flip selected bits.
      u = bitxor(u,2.^(b-1));
      y(k) = mysign(y(k)).*u; 
   end

end
