# Introduction to GitHub and Getting Setup

## Getting Started with GitHub

* [Version Control with Git](http://faculty.smu.edu/csc/workshops/2020/summer/git/)
* [Getting Started with GitHub](https://docs.github.com/en/github/getting-started-with-github)

## Initial Setup of Your Repository and Python Environment

1. Go to the [GitHub repository for our class Jupyter notebooks](https://github.com/SouthernMethodistUniversity/ds_1300_notebooks).
2. Select "Fork". This will create a personal copy of the repository.
3. Select "Code" and copy the HTTPS URL for the forked repository.
4. Go to the [HPC Portal](https://hpc.smu.edu) and log in.
5. Select "Interactive Apps" and then "JupyterLab".
6. Set the fields to:
  * "Additional environment modules to load": `python/3`
  * "Custom module paths": Clear the contents if any
  * "Custom environment settings": Clear the contents if any
  * "Partition": `htc`
  * "Number of hours" `1`
  * "Number of nodes": `1`
  * "Cores per node": `1`
  * "GPUs per node": `0`
  * "Memory": `6`
7. Select "Launch" and wait for the job and JupyterLab session to begin (this can take a few minutes).
8. Select "Connect to JupyterLab", which will appear when the JupyterLab session has started.
9. Select "File", "New", "Terminal" to start a new terminal session within JupyterLab.
10. In the terminal session, type each command and press Enter:

```
git clone <paste the URL for the forked repository without brackets>
cd ds_1300_notebooks
git remote add upstream https://github.com/SouthernMethodistUniversity/ds_1300_notebooks.git
mkdir work
module load python/3
conda env create -f environment.yml --force
```

## Launching a JupyterLab Session and Accessing Your Notebooks

1. Go to the [HPC Portal](https://hpc.smu.edu) and log in.
2. Select "Interactive Apps" and then "JupyterLab".
3. Set the fields to:
  * "Additional environment modules to load": `python/3`
  * "Custom module paths": Clear the contents if any
  * "Custom environment settings": `source activate ~/.conda/envs/ds_1300`
  * "Partition": `htc`
  * "Number of hours" `24`
  * "Number of nodes": `1`
  * "Cores per node": `1`
  * "GPUs per node": `0`
  * "Memory": `6`
4. Select "Launch" and wait for the job and JupyterLab session to begin (this can take a few minutes).
5. Select "Connect to JupyterLab", which will appear when the JupyterLab session has started.
6. Select the "File Browser" icon at the top of the left sidebar and browse to your `ds_1300_notebooks` directory.

## Working on Notebooks

1. Browser into "code" directory.
1. Right-click on a notebook and select "Copy".
2. Browse into your "work" directory.
3. Right-click with the directory and select "Paste".
4. Double click the notebook to open.

## Pulling New Notebooks

1. Save the notebook.
2. Open a terminal session via "File", "New", "Terminal".
3. In the terminal session, type each command and press Enter after each line:

```
cd ds_1300_notebooks
git fetch upstream
git merge upstream/main
```

```{note}
If you have edited a file tracked upstream, *e.g.*  `code/01_assignment.ipynb`,
you won't be able to merge. Checkout the file to clear the changes, *e.g.* `git
checkout code/01_assignment.ipynb`, and then proceed with `git merge
upstream/main`.
```

## Submitting Completed Assignments via GitHub

1. Save the notebook.
2. Open a terminal session via "File", "New", "Terminal".
3. In the terminal session, type each command and press Enter after each line:

```
cd ds_1300_notebooks
git add work/<the name of the completed notebook without brackets>.ipynb
git commit -m "<Brief note about work witout brackets>"
git push
git tag -a <assignment number without brackets> -m "<Completed assignment number without brackets>"
git push --tags
```
## Add Dr. Godat and Dr. Kalescky to Your Forked GitHub Repository

1. Log into GitHub
2. Go to your ds_1300_notebooks repository
3. Select "Settings"
4. Select "Manage access" on the left-side panel
5. Select "Invite a collaborator"
6. Type "egodat" in the field and then select "Eric Godat" when it appears
7. Select "Add egodat to this repository"
8. Select "Invite a collaborator"
9. Type "rkalescky" in the field and then select "Robert Kalescky" when it appears
10. Select "Add rkalescky to this repository"
