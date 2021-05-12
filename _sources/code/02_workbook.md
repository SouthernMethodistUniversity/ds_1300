---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Working with Data


By: Dr. Eric Godat and Dr. Rob Kalescky 


As implied by the name, a Data Scientist needs to be able to work with data. However, what consitutes data can vary wildly depending on the project you're working on.

In this notebook, we will dive into a few common types of data and some of the common pitfalls you'll encounter.

```python
import pandas as pd
```

## Loading Data into Python


The first step is getting data into python. While you could type the data into a dictionary, list, or other data format, that quickly becomes unsustainable. Fortunately there are several ways to load our data directly. 


### From csv


The easiest way to load data is to use pandas to [read a csv](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) (comma separated values) file into a data frame. This also works with other deliminators (things that split your data fields) too.

```python
df = pd.read_csv("../data/sample_data.csv")
df
```

## Numbers as Data


The most classic example of data (and the one most people think of when you say data) is numerical data.


Using the data we just loaded. Let's do some basic calculations.

We're going to start by summing our data.

```python
df.sum()
```

> Obviously this doesn't make any sense, but why doesn't it?


How about the difference in price between two houses?

```python
df['price_2'][0]-df['price_1'][0]
```

## Text as Data

```python
df = pd.read_csv("../data/folktales.csv")
```

```python
df
```

## People as Data

```python
pd.read_csv("../data/messy_data.csv",delimiter=";")
```

## Common Issues


### Names and Unique Identifiers


### Dates


### Missing Values


## Getting Help


### Python Documentation


### Stack Overflow, Stack Exchange, and more

```python

```
