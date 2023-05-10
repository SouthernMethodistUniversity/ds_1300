# DS1300: A Practical Introduction to Data Science

We live in a world driven by computers, algorithms, and data. This
course seeks to equip students with the basic knowledge and skills to
not only understand but to use the tools needed to model and make
decisions about the world around them. Students will be introduced to
basic programming in Python, high-performance computing using ManeFrame
II, and data science.

# Using Anaconda

## Installing and Loading Anaconda

This step can be skipped when using SMU HPC clusters as Anaconda is already
installed and available via `module load conda`.

Directions for installing Anaconda on your own machine can be found
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
Either the full Anaconda or the miniconda distributions will work. The
differences between the two are noted on the installation page. More
information on getting started can be found
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda).

## Installing the `ds_1300` Environment

1. `mamba env create -f environment.yml --force`

## Using the `ds_1300` Environment

1. `mamba activate ds_1300` to start (`source activate ~/.conda/envs/ds_1300`
   on M2)
2. `mamba deactivate` when finished

## Remove Environment

1. `mamba remove --name ds_1300 --all`

# Using Python Environments

## Loading a Python Environment

There are many Python 3 installations on SMU HPC clusters. An example from M3
is given:

```
module purge
module load gcc python
```

## Installing the `ds_1300` Environment

```
python3 -m venv ~/.venv/ds_1300
source ~/.venv/ds_1300/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

## Using the `ds_1300` Environment

1. `source ~/.venv/ds_1300/bin/activate` to start
2. `deactivate` when finished

## Remove Environment

1. `rm -rf ~/.venv/ds_1300`

