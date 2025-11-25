Stages of PRF scotoma analyses:

Run preprocessing: use JH linescanning pipeline (see https://github.com/gjheij/linescanning/tree/main/linescanning)


[s1] psc_time_series
* For each task & sub, average time courses across runs (1,2); 
* Note baselining to 0, based on first 19 TRs... 

[s2] call_G_fit_data
* Fit the gaussian 

[s3] call_N_fit_data
* Fit extended (normalisation) models to the data
* HRF is fixed for each voxel, based on [s2a]

