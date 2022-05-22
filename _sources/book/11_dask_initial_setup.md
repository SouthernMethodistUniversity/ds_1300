# Getting Started

## Launch Multicore Dask Enabled JupyterLab Environment

1. Go to the [HPC Portal](https://hpc.smu.edu) and log in.
2. Select "Interactive Apps" and then "JupyterLab".
3. Set the fields to:
   * "Additional environment modules to load": `python/3`
   * "Custom module paths": Clear the contents if any
   * "Custom environment settings": `source activate ~/.conda/envs/ds_1300`
   * "Partition": Select any appropriate partition other than `htc`; recommend `development` or `standard-mem-s`
   * "Number of hours" `2`
   * "Number of nodes": `1`
   * "Cores per node": `4`
   * "GPUs per node": `0`
   * "Memory": `24`
4. Select "Launch" and wait for the job and JupyterLab session to begin (this can take a few minutes).
5. Select "Connect to JupyterLab", which will appear when the JupyterLab session has started.

