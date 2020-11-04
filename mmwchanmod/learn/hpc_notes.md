# Running Training in the NYU HPC cluster

The model training can be long.  This note provides instructions to run the training in NYU HPC's cluster.  The instructions can be modified for other clusters that use the [slurm utilty](https://slurm.schedmd.com/tutorials.html).

## Getting an HPC account
Before beginning, you will need an account on NYU's HPC cluster.  Follow the [instructions](https://www.nyu.edu/life/information-technology/research-and-data-support/high-performance-computing.html) on the NYU HPC website.

## Loading python and tensorflow

* Once you have an account, log into one of the cluster machines such as `prince.hpc.nyu.edu`.
* On that remote machine, you will first need to load python3.  Find the latest version of python3 with the command
    ```
        module avail python3
    ```
    Assuming the latest version is `python3/intel/3.7.3`, we load:
    ```
        module load python3/intel/3.7.3
    ```
*  The first time you log in, you will also need to install `sklearn` and `tensorflow`:
    ```
        pip3 install --upgrade --user sklearn
        pip3 install --upgrade --user tqdm
        pip3 install --upgrade --user tensorflow       
    ```
    Note that earlier versions of tensorflow had separate GPU and CPU versions.  This is no longer necessary.  The `--user` option is needed to provide you the permissions.
* If you are running on a GPU, you will need to install the CUDA drivers (these commands also need to be run every time you log in).  First find the latest version with `cudnn` drivers with `module avail cudnn`.  Assuming the latest driver is:   `cudnn/10.1v7.6.5.32`
    ```
        module load cudnn/10.1v7.6.5.32
    ```
    Note that the networks in this project are small and a GPU will not offer any noticeable improvement in speed. 

## Running the training on the cluster
To run the training on the cluster:
* SSH to prince.hpc.nyu.edu
* Download the repository at:
    ```
        git clone https://github.com/sdrangan/uavchanmod.git
    ```
   This will download the repository to the directory `uavchanmod`.
* Change directory `cd uavchanmod`.
* The script file `hpc_train.sh` is the script that will run the command `train_mod.py` in the cluster.  You may wish to change some of the parameters such as the notification email and output directory. You can also change parameters for the `train_mod`.
* Once you have modified `hpc_train.sh` you can run the script in the cluster with:
    ```
        sbatch hpc_train.sh
    ```
* After the job is complete, the outputs will be in the model directory that was specified in the `--model_dir` option.  



## Get access to Google drive
You will likely want to copy the model files back to your local machine where you can perform the plotting.  The NYU HPC has a nice utility, `rclone`, from which we can directly transfer files between the HPC cluster and Google drive.  To use the utility, first find the most recent version of the module with the command,
```
    module avail rclone
```
This will list all the modules available and their versions.  Assuming the most recent version is 1.50.2, we can load the `rclone` module with the command:
```
    module load rclone/1.50.2
```
You will have to set up the `rclone` module to connect to your Gdrive.  Type,
```
    rclone config
```
and then follow the instructions to create a new remote.  This [website](https://rclone.org/drive/) has an example of how you
should fill in the commands.  For the remainder of the instructions, we will assume that we called the drive `gdrive`.

Once you configure your Gdrive, you can see the drive in
```
    rclone config
```
You should see `gdrive` drive. You can see directories with commands such as
```
    rclone lsd gdrive:/directory_name
```
You can copy the `model_data` with commands like:
```
    rclone copy model_data gdrive:/destpath/model_data
```
The files should now be in your Gdrive.






