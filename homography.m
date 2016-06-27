% homography from the dvs to the corresponding IDS
% the solver is according to https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
function H = homography(pt_dvs, pt_ids)
%% pts_dvs, pts_ids are NX2 matrics, with (xi, yi)...


N = size(pt_dvs,1);

A = zeros(2*N, 9);

for i = 1:N
    A(2*i-1,:) = [-pt_dvs(i,1),-pt_dvs(i,2), -1, 0,0,0,pt_ids(i,1)*pt_dvs(i,1), pt_ids(i,1)*pt_dvs(i,2),pt_ids(i,1) ];
    A(2*i,:) = [0,0,0, -pt_dvs(i,1),-pt_dvs(i,2), -1, pt_ids(i,2)*pt_dvs(i,1), pt_ids(i,2)*pt_dvs(i,2),pt_ids(i,2) ];
end


[V,D] = eig(A'*A);



hh_opt = V(:,1);

H = (reshape(hh_opt, [3,3]))';

end

