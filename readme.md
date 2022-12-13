## Usage of the program

### read imagestack

call readfrogs() and change the file name inside

### Compute and store the shadow time and shadow line

main function process()
uncomment 
```
# t_sha = compute_shadow_time()
# showimage(t_sha, 'jet')
# plt.imsave('shadow_time.jpg', t_sha, cmap = 'jet')
# np.save('shadow_time.npy', t_sha)
```

### calibration to compute the intrinsic and extrinsic calibration

call calib()

### reconstruction based on intrinsic, extrinsic calibration and shadow line and time

call reconstruct()

### display the 3D point clouds 

call showscatter(plist, flist), where plist is a N x 3 numpy array 
of (x,y,z) coordinates, flist is the corresponding N x 1 intensity values 

### npz files

shadow plane parameters in shadow_plane.npz, format:

```t_list``` does not include all the frames, it only includes the index value of all the 
frames that actually generates valid shadow line and planes.(55-130th frame approximately)

```
ext_out = {"frame_t": np.array(t_list), "p1":np.array(p1_list), "n":np.array(n_list)}
```



reconstructed 3D points request in stored in reconstruct_points.npz

It also has the same format like the shadow plane npz files.(The first dimension is the valid frame list)
