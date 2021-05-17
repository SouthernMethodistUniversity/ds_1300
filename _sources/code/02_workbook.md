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
df = pd.read_csv("../data/sample_data.csv") # This means that our data lives one level up (..) and in a directory named data
df
```

## Numbers as Data


The most classic example of data (and the one most people think of when you say data) is numerical data.

Using the data we just loaded. Let's ask a question and work towards a solution.

Before we start, let's look at two neat tricks that will come in handy as we explore our data.

```python
# Trick 1: Getting a list of columns names
df.columns
```

```python
# Trick 2: Slicing multiple columns
df[['Property','bedrooms','bathrooms']] # Rember that a single [] will give us a single column. Using [[]] lets us select multiple columns
```

```python
# Trick 3: Only showing a few lines
df.head(3)
```

```python
df.tail(3)
```

### Question 1) Which property increased in value the most from price 1 to price 3?

Let's assume that our data means price in year 1, year 2, year 3. This is a guess we have to make because we don't know much about where this data comes from.



Now that we have a little less to look at, we want to make a column with the increase in price.

```python
df['increase'] = df['price_3']-df['price_1'] #operation on 2 columns, saving to a new column
```

```python
df
```

```python
df.sort_values(by='increase',ascending=False) #Sort values by the column 'increase', we want the largest values at the top so ascending needs to be false
```

Answer 1) Property F has increased the most in value


### Question 2) Which property is the best price (year_3) per square foot (sqft)?

```python
df['p/sqft']=df['price_3']/df['sqft']
```

```python
df.sort_values(by='p/sqft',ascending=False)
```

Answer 2) Property C has the best price per square foot


### Question 3) I'm in the market for a house that has more than 3 bedrooms and more than 2 bathrooms, what are my options?

```python
df[(df['bedrooms']>3)&(df['bathrooms']>2)]
```

Answer 3) B or C are good options for what I'm looking for


### Question 4) I'm a relator and trying to write a formula for the best house to show my clients. I want to show them the place with high bed/bath/sqft with a low price but consistent growth. How could I make that determination?


Let's break this one down into smaller pieces. I need to:
- Combine bedrooms, bathrooms, and sqft into a single number (larger is better)
- Factor in price (smaller is better)
- Consistent growth, maybe this could be an average of price_1 -> price_2 and price_2 -> price_3 ? Maybe we need to think about this one.
- Combine all of this into a single score
- Finally sort and cut the results down for the client

```python
df['bbsqft'] = (df['bedrooms']+df['bathrooms'])*df['sqft']
```

```python
df['bbsqft/p3'] = df['bbsqft']/df['price_3']
```

```python
df['p2-p1'] = df['price_2']-df['price_1']
```

```python
df['p3-p2'] = df['price_3']-df['price_2']
```

```python
df['growth_1'] = (df['p3-p2']+df['p2-p1'])/2
```

```python
#This gives us a boolean column. Booleans can act like 1 (True) and 0 (False) if we want to use them in calculations
df['growth_2'] = df['p3-p2']>=df['p2-p1']
#This will let us zero out scores that don't meet our criteria
```

```python
df['score'] = 100*df['bbsqft/p3']*df['growth_1']*df['growth_2'] # Added a scaling factor to make the numbers easier
```

```python
df[['Property','price_3','bedrooms','bathrooms','score']].sort_values(by='score',ascending=False)
```

If I find out later that my client has a budget of 600, can I adapt my data to only show them those?

```python
df[df['price_3']<600]
```

```python
df[df['price_3']<600][['Property','price_3','bedrooms','bathrooms','score']].sort_values(by='score',ascending=False)
```

Looks like I should start by showing my client property A

**Is this the only way I could do this? What could we change?**


## Text as Data


A type of data that has become especially popular and powerful to investigate is text. Turns out there is a lot that we can learn by looking at what we write down. We'll spend more time working with text later in the class but for now, we'll just load the data and do some basic parsing.

```python
df = pd.read_csv("../data/folktales.csv")
```

```python
df
```

### Question 1) What countries do we have stories from?

```python
df['Country of Origin'].values
```

```python
# How about a more compressed list
df['Country of Origin'].value_counts()
```

### Question 2) What fraction of my stories were written by the Brothers Grimm?

```python
total_stories = len(df)
total_stories
```

```python
grimm = len(df[(df['Author'].str.contains('Grimm'))==True])
grimm
```

```python
grimm/total_stories
```

### Question 3) How many titles contain animals?


How would I even do this?

```python
df['Title'].str.contains('animals').value_counts()
```

That clearly doesn't seem like what the question is asking... Maybe this isn't something we can answer. Why not? What would we need to answer this question?


## People as Data


Another common type of data set is personal information. Just think of every sign up sheet, grade book, or class roster. One major problem with personal data is that people generally don't fit into clean data "boxes".

```python
messy = pd.read_csv("../data/messy_data.csv",delimiter=";")
messy
```

Let's just look at this data. How many data issues can you find that would hinder an analysis? How would you handle it?


## Tricks for messy data


#### Names


One common problem is that names tend to be really bad ways to identify people. Why is that?

A solution to this is to use something called a unique identifier (think your SMU ID number). A unique identifier can be used instead of a name because it will have a standard format and generally can be used to link an individual across multiple data sets. If used properly it can also be a good way to de-identify individuals.

```python
import random
messy['uID'] = [str(random.randint(0,1000)).zfill(4) for i in range(len(messy))]
messy
```

There is a lot to unpack with what we did there, let's break that down:

- First we're generating a list using a single line for loop
- Our loop is over the elements in the range that goes from 0 to the length of our data frame - effectively saying make the list the same length as our dataframe
- Then our loop generates a random integer from 0 to 1000 but we needed to import python's random number generator to do that for us
- Then we want to convert our random integer to a string. We wouldn't want to accidentally do math with our unique ID numbers
- We want to make sure our IDs are all the same length using zfill to add 0s to the front of our string. This is common for numbers like this. Just think of your credit card number, social security number, SMU ID....
- Lastly, we assign our list to the new column in our dataframe 'uID'

```python
# The same code but unpacked
import random

ll = [] #initialize an empty list
length = len(messy)
for i in range(0,length):
    r = random.randint(0,1000)
    s = str(r)
    s4 = s.zfill(4)
    ll.append(s4) # This lets us add elements to a list
messy['uID']=ll
messy
```

#### Dates


Another common issue is that there are lots of formats for dates and times. This isn't just an issue with personal data but is one that can cause huge headaches when working with data sets. Even asking simple questions can become complicated when working with dates if you aren't sure of the formatting.

Think about all the steps your brain makes if I ask you what the date was 3 weeks ago?


Somewhat incredibly, pandas can actually recognize several forms of dates and guess what the date formats are and convert them to a standardize format using the function [to_datetime](https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html?highlight=to_datetime#pandas.to_datetime).

```python
messy['Date_Fixed'] = pd.to_datetime(messy['Air_Date'],errors='raise')
messy[['Air_Date','Date_Fixed']]
```

#### Missing Values


Another problem we can encounter is missing data. This happens all the time with "wild data" and can happen for numerous reasons, for example:
- no data should exist for a reason
- an error in the data creation
- an operation induced the missing data

Pandas has a simple function to handle this called [fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html).

```python
messy = messy.fillna('No Data Available') #Note this will change our data
messy
```

#### Case Sensitivity


Another common issue with dealing with messy data is case sensitivity. Since python sees 'A' and 'a' as two different characters, it is important to be aware of case sensitivity. The easiest way to do this is to send all the characters in a particular column to a single case pattern. Fortunately, pandas has a family of functions to do that for us. [lower](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.lower.html?highlight=lower#pandas.Series.str.lower) is a good example.

```python
messy['Ship'] = messy['Ship'].str.lower()
messy
```

That might not actually be what we want but at least we can compare the values now.

```python
messy['Ship'].value_counts()
```

### Wrap Up


Let's look at our messy data before:

```python
before = pd.read_csv("../data/messy_data.csv",delimiter=";")
before
```

and after:

```python
messy
```

What were we able to fix? What else could we do?

```python

```
