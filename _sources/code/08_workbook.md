---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Data Storage

Efficient storage can dramatically improve performance, particularly when operating repeatedly from disk.

Decompressing text and parsing CSV files is expensive.  One of the most effective strategies with medium data is to use a binary storage format like HDF5.  Often the performance gains from doing this is sufficient so that you can switch back to using Pandas again instead of using `dask.dataframe`.

In this section we'll learn how to efficiently arrange and store your datasets in on-disk binary formats.  We'll use the following:

1.  [Pandas `HDFStore`](http://pandas.pydata.org/pandas-docs/stable/io.html#io-hdf5) format on top of `HDF5`
2.  Categoricals for storing text data numerically

**Main Take-aways**

1.  Storage formats affect performance by an order of magnitude
2.  Text data will keep even a fast format like HDF5 slow
3.  A combination of binary formats, column storage, and partitioned data turns one second wait times into 80ms wait times.


## Read CSV


First we read our csv data as before.

CSV and other text-based file formats are the most common storage for data from many sources, because they require minimal pre-processing, can be written line-by-line and are human-readable. Since Pandas' `read_csv` is well-optimized, CSVs are a reasonable input, but far from optimized, since reading required extensive text parsing.

```python
import os
filename = os.path.join('/hpc/classes/ds_1300_data', 'accounts.*.csv')
filename
```

```python
import dask.dataframe as dd
df_csv = dd.read_csv(filename)
df_csv.head()
```

### Write to HDF5


HDF5 and netCDF are binary array formats very commonly used in the scientific realm.

Pandas contains a specialized HDF5 format, `HDFStore`.  The ``dd.DataFrame.to_hdf`` method works exactly like the ``pd.DataFrame.to_hdf`` method.

```python
target = os.path.join('../work', 'accounts.h5')
target
```

```python
# convert to binary format, takes some time up-front
%time df_csv.to_hdf(target, '/data')
```

```python
# same data as before
df_hdf = dd.read_hdf(target, '/data')
df_hdf.head()
```

### Compare CSV to HDF5 speeds


We do a simple computation that requires reading a column of our dataset and compare performance between CSV files and our newly created HDF5 file.  Which do you expect to be faster?

```python
%time df_csv.amount.sum().compute()
```

```python
%time df_hdf.amount.sum().compute()
```

Sadly they are about the same, or perhaps even slower. 

The culprit here is `names` column, which is of `object` dtype and thus hard to store efficiently.  There are two problems here:

1.  How do we store text data like `names` efficiently on disk?
2.  Why did we have to read the `names` column when all we wanted was `amount`


### 1.  Store text efficiently with categoricals


We can use Pandas categoricals to replace our object dtypes with a numerical representation.  This takes a bit more time up front, but results in better performance.

More on categoricals at the [pandas docs](http://pandas.pydata.org/pandas-docs/stable/categorical.html) and [this blogpost](http://matthewrocklin.com/blog/work/2015/06/18/Categoricals).

```python
# Categorize data, then store in HDFStore
%time df_hdf.categorize(columns=['names']).to_hdf(target, '/data2')
```

```python
# It looks the same
df_hdf = dd.read_hdf(target, '/data2')
df_hdf.head()
```

```python
# But loads more quickly
%time df_hdf.amount.sum().compute()
```

This is now definitely faster than before.  This tells us that it's not only the file type that we use but also how we represent our variables that influences storage performance. 

How does the performance of reading depend on the scheduler we use? You can try this with threaded, processes and distributed.

However this can still be better.  We had to read all of the columns (`names` and `amount`) in order to compute the sum of one (`amount`).  We'll improve further on this with `parquet`, an on-disk column-store.  First though we learn about how to set an index in a dask.dataframe.


### Exercise


`fastparquet` is a library for interacting with parquet-format files, which are a very common format in the Big Data ecosystem, and used by tools such as Hadoop, Spark and Impala.

```python
target = os.path.join('../work', 'accounts.parquet')
df_csv.categorize(columns=['names']).to_parquet(target, storage_options={"has_nulls": True}, engine="fastparquet")
```

Investigate the file structure in the resultant new directory - what do you suppose those files are for?

`to_parquet` comes with many options, such as compression, whether to explicitly write NULLs information (not necessary in this case), and how to encode strings. You can experiment with these, to see what effect they have on the file size and the processing times, below.

```python
ls -l ../work/accounts.parquet/
```

```python
df_p = dd.read_parquet(target)
# note that column names shows the type of the values - we could
# choose to load as a categorical column or not.
df_p.dtypes
```

Rerun the sum computation above for this version of the data, and time how long it takes. You may want to try this more than once - it is common for many libraries to do various setup work when called for the first time.

```python
%time df_p.amount.sum().compute()
```

When archiving data, it is common to sort and partition by a column with unique identifiers, to facilitate fast look-ups later. For this data, that column is `id`. Time how long it takes to retrieve the rows corresponding to `id==100` from the raw CSV, from HDF5 and parquet versions, and finally from a new parquet version written after applying `set_index('id')`.

```python
# df_p.set_index('id').to_parquet(...)
```

Adapted from [Dask Tutorial](https://tutorial.dask.org).

