# Introduction to GitHub

## Getting Started

* [Version Control with Git](http://faculty.smu.edu/csc/workshops/2020/summer/git/)
* [Getting Started with GitHub](https://docs.github.com/en/github/getting-started-with-github)

## Setting Up Your Repository and Python Environment

#. Go to the [GitHub repository for our class Jupyter notebooks](https://github.com/SouthernMethodistUniversity/ds_1300_notebooks).
#. Select Fork. This will create a personal copy of the repository.
#. Select Code and copy the HTTPS URL for the forked repository.
#. Go to the [HPC Portal](https://hpc.smu.edu) and log in.
#. Select Clusters and then ManeFrame II Shell Access.
#. In the terminal session, type each command and press Enter:

```
git clone <paste the URL for the forked repository without brackets>
cd ds_1300_notebooks
module load python/3
conda env create -f environment.yml --force
```

