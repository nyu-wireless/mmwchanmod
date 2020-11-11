#  Generative Neural Networks for 28 GHz UAV Air-to-Ground Channels

* Sundeep Rangan, William Xia, Marco Mezzavilla, Giuseppe Loianno (NYU)
* Giovanni Geraci, Angel Lozano (UPF, Barcelona)
* Vasilii Semkin (VTT, Finland)

This project utilizes the tools in the repository for training
28 GHz air-to-ground channel models.  

A full paper describing the work can be found in:

* Xia, W., Rangan, S., Mezzavilla, M., Lozano, A., Geraci, G., Semkin, V., & Loianno, G. (2020). Millimeter Wave Channel Modeling via Generative Neural Networks. arXiv preprint arXiv:2008.11006.


## Pre-Trained Models
Air-to-ground channel models at 28 GHz are trained for four sections of the following cities:

* London and Tokyo (links in the two cities are combined)
* Boston
* Beijing
* Moscow

The repository provides methods to directly access the pre-trained models.
We will upload a demonstration shortly, but you can see how to use
the models from the examples in the code below.


## Training the Models from Scratch
The data used to train the model can be downloaded following the
[dataset demo](../../demos/dataset_demo.ipynb).  Feel free to use this
for other projects.

If you wish to re-train the models from scratch, first run the command:
```
   python uav_download.py    
```
This will download the London and Tokyo UAV datasets to the
local machine and create a concatanated dataset.  Then run the commands
```
   python train_mod.py --ds_name uav_lon_tok --model_dir ../../models/uav_lon_tok
```
The command `train_mod` has commands to change the number of epochs,
and model parameters.  You can also train the other models,
`uav_boston`, `uav_moscow` and `uav_beijing`.  

Training requires a large number of epochs (5000), so you will likely 
want to do this on a cluster.  If you are using a slurm utility,
you can use the batch file, `hpc_train.sh`. 


## Plot the Omni-directional Path Loss and RMS Delay Spread
To evaluate the trained models, the project compares the conditional
distribution of various channel statistics on the test data
with the conditional distribution generated from the trained models.

For example, we can compare the omni-directional path loss with the
command:
```
    python plot_path_loss_cdf.py --model_city LonTok --plot_fn pl_cdf_lt.png
```
This command will plot the CDF of the omni-directional path loss for the London-Tokyo data.
It will also plot the CDF of the points generated from the model trained
on the London-Tokyo data.  The two curves should match.
```
    python plot_path_loss_cdf.py --model_city "LonTok Moscow Beijing Boston"
        --plot_fn pl_cdf_all.png
```
This command will also plot the CDF of points generated from models trained
in other environments.  You will see a signficant discrepancy, particularly
in the aerial cells.

You can similarly plot the RMS delay spread with:
```
    python plot_dly_cdf.py --model_city LonTok --plot_fn dly_cdf_lt.png
    python plot_dly_cdf.py --model_city "LonTok Moscow Beijing Boston"
        --plot_fn dly_cdf_all.png
```


## Plotting the Angle Distribution

You can plot the distribution of the angles of arrival and departure
as a function of the distance from the test data and model with:
```
    python plot_angle_dist.py --mod_name uav_lon_tok --ds_name uav_lon_tok
        --plot_fn angle_dist_lt.png
```
   
## Acknowledgments
W.  Xia,  M.  Mezzavilla  and  S.  Rangan  were  supportedby  NSF  grants  1302336,  1564142,  1547332,  and  1824434,NIST, SRC, and the industrial affiliates of NYU WIRELESS.A.  Lozano  and  G.  Geraci  were  supported  by  the  ERC  grant694974,  by  MINECO’s  Project  RTI2018-101040,  and  by  the Junior Leader Program from "la Caixa" Banking Foundation.

All authors are also grateful for the assistance from Remcom that provided
the Wireless Insite tool to generate the data.
