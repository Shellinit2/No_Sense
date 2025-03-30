# No_Sense
Realsense is so too expensive INTEL

So, I want a realsense but its 50k so fuck you intel…



The plan is to create a sterio camera (RGBD) and then convert that into a point cloud

And then put an IMU on it to calculate optical flow.



How does sterio work:
So there are 2 cameras looking at the same thing. parallel system ofcourse (for now for sure, lucius fox … too much power for one person my ass). 
There will be a little bit of difference in the feature’s pixel position in the images. This is called disparity. And this is directly proportional to the distance of the object. 


		



now we need to find a matching window from both the images so that we can calculate the disparity.
lets consider a fixed window size - 5×5 or 7×7 and divide the images into blocks. All we have to do is to just go on each horizontal row of blocks (cause the images will be perfectly Rectified cause they are from a parallel system) and find out which block has the least difference





$SAD[sum of squared differences]= \text{SAD}(d) = \sum_{i,j} \left| I_L(x + i, y + j) - I_R(x + d + i, y + j) \right|$

$SSD[sum of absolute diff]= \text{SSD}(d) = \sum_{i,j} \left( I_L(x + i, y + j) - I_R(x + d + i, y + j) \right)^2$

{Doubt: what if there is a monochromatic background? will it choose the disparity from the the sky???}



The disparity that minimizes the matching cost (SAD or SSD) is the best match for that pixel/block.

Once this is done disparity map is generated. And then post processing like median or bilateral filtering is applied

once this is done… depth of each pixel is 
$Z=(f*B)/d$



For a scene where objects are relatively far away, a maximum disparity of 16-64 might be sufficient. For closer objects (such as in indoor environments) use higher disparity values (like 128) to capture the large disparities that occur at short distances.



3d reconstruction:
	$X=(x-cx)*Z/fx$
	$Y=(y-cy)*Z/fy$
	$Z=(f*B)/d$


Rotation matrix for quaternion 



$R=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$

​

$q = \frac{1}{4q_w}(q_x, q_y, q_z, q_w)$



$q_w = sqrt(1 + r_{11} + r_{22} + r_{33})/{2}$

$q_x = \frac{r_{32} - r_{23}}{4q_w} $

 $q_y = \frac{r_{13} - r_{31}}{4q_w} $

 $q_z = \frac{r_{21} - r_{12}}{4q_w}$




##############################################################################
Run each script

point_cloud_rgbd_opticalflow.py
point_cloud_gen.py
sensor_publisher.py
