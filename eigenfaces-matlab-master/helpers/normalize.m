function [T2] = normalize(T,m,S)
    %normalize: normalise image vector, using means and std.
    % usage:  [T2] = normalize(T,m,S)
    %
    % where,
    % ARGS:
    %    T is the image array. (it could be one or several images, in
    %    vectors)
    %    m is the mean of T.
    %    S is the result of the std fonction on T.
    % RETURNS:
    %    T2 is T normalized.
    
    % './' : division point by point
    T2 = (T - m) ./ S;
end

