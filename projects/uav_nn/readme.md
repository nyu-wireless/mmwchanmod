#  Generative Neural Networks for 28 GHz UAV Air-to-Ground Channels

* Sundeep Rangan, William Xia, Marco Mezzavilla, Giuseppe Loianno (NYU)
* Giovanni Geraci, Angel Lozano (UPF, Barcelona)
* Vasilii Semkin (VTT, Finland)

This project utilizes the tools in the repository for training
28 GHz air-to-ground channel models.  

A full paper describing the work can be found in:

* Xia, W., Rangan, S., Mezzavilla, M., Lozano, A., Geraci, G., Semkin, V., & Loianno, G. (2020). Millimeter Wave Channel Modeling via Generative Neural Networks. arXiv preprint arXiv:2008.11006.


## Downloading the Data
You should first download the data:
```
   python uav_download.py    
```
This will download the London and Tokyo UAV datasets and create a concatanated dataset.


## Plot the omni-directional path loss
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
   
## Acknowledgments
W.  Xia,  M.  Mezzavilla  and  S.  Rangan  were  supportedby  NSF  grants  1302336,  1564142,  1547332,  and  1824434,NIST, SRC, and the industrial affiliates of NYU WIRELESS.A.  Lozano  and  G.  Geraci  were  supported  by  the  ERC  grant694974,  by  MINECO’s  Project  RTI2018-101040,  and  by  the Junior Leader Program from "la Caixa" Banking Foundation.

All authors are also grateful for the assistance from Remcom that provided
the Wireless Insite tool to generate the data.
