function [n,V,p] = ComputePlane(I1, I2, stereoParams)
%input: (1) two images contains the plane; (2) calibration data
%output: the plane in the 3D space, the coordinate is the same with camera1
    
    %based on the following work: >>>
    %Computes the plane that fits best (lest square of the normal distance
    %to the plane) a set of sample points.
    %INPUTS:
    %
    %X: a N by 3 matrix where each line is a sample point
    %
    %OUTPUTS:
    %
    %n : a unit (column) vector normal to the plane
    %V : a 3 by 2 matrix. The columns of V form an orthonormal basis of the
    %plane
    %p : a point belonging to the plane
    %
    %NB: this code actually works in any dimension (2,3,4,...)
    %Author: Adrien Leygue
    %Date: August 30 2013
    % <<<
    
    
    % image undistortion 
    I1 = undistortImage(I1, stereoParams.CameraParameters1);
    I2 = undistortImage(I2, stereoParams.CameraParameters2);
    
    % corresponding detection
    % here we select the control points
    [movingPoints, fixedPoints]=cpselect(I1, I2, 'Wait', true);
    % points triangulation
    worldPoints = triangulate(movingPoints,fixedPoints,stereoParams);
    
    figure;plot3(worldPoints(:,1),worldPoints(:,2),worldPoints(:,3),'ro');
    
    % plane fitting  
    %the mean of the samples belongs to the plane
    p = mean(worldPoints,1);
    
    %The samples are reduced:
    R = bsxfun(@minus,worldPoints,p);
    %Computation of the principal directions if the samples cloud
    [V,D] = eig(R'*R);
    %Extract the output from the eigenvectors
    n = V(:,1);
    V = V(:,2:end);
end