# Introduction to GitHub and Getting Setup

## Getting Started with GitHub

* [Version Control with Git](http://faculty.smu.edu/csc/workshops/2020/summer/git/)
* [Getting Started with GitHub](https://docs.github.com/en/github/getting-started-with-github)

## Initial Setup Your Repository and Python Environment

#. Go to the [GitHub repository for our class Jupyter notebooks](https://github.com/SouthernMethodistUniversity/ds_1300_notebooks).
#. Select "Fork". This will create a personal copy of the repository.
#. Select "Code" and copy the HTTPS URL for the forked repository.
#. Go to the [HPC Portal](https://hpc.smu.edu) and log in.
#. Select "Interactive Apps" and then "JupyterLab".
#. Set the fields to:
  * "Additional environment modules to load": `python/3`
  * "Custom module paths": Clear the contents if any
  * "Custom environment settings": Clear the contents if any
  * "Partition": `htc`
  * "Number of hours" `1`
  * "Number of nodes": `1`
  * "Cores per node": `1`
  * "GPUs per node": `0`
  * "Memory": `6`
#. Select "Launch" and wait for the job and JupyterLab session to begin (this can take a few minutes).
#. Select "Connect to JupyterLab", which will appear when the JupyterLab session has started.
#. Select "File", "New", "Terminal" to start a new terminal session within JupyterLab.
#. In the terminal session, type each command and press Enter:

```
git clone <paste the URL for the forked repository without brackets>
cd ds_1300_notebooks
module load python/3
conda env create -f environment.yml --force
```

## Launching a Jupyter Lab Session and Accessing Your Notebooks

#. Go to the [HPC Portal](https://hpc.smu.edu) and log in.
#. Select "Interactive Apps" and then "JupyterLab".
#. Set the fields to:
  * "Additional environment modules to load": `python/3`
  * "Custom module paths": Clear the contents if any
  * "Custom environment settings": `source activate ~/.conda/envs/ds_1300`
  * "Partition": `htc`
  * "Number of hours" `1`
  * "Number of nodes": `1`
  * "Cores per node": `1`
  * "GPUs per node": `0`
  * "Memory": `6`
#. Select "Launch" and wait for the job and JupyterLab session to begin (this can take a few minutes).
#. Select "Connect to JupyterLab", which will appear when the JupyterLab session has started.
#. Select the "File Browser" icon at the top of the left sidebar and browse to your `ds_1300_notebooks` directory.

## How to Submit Assignments via GitHub


