# Intermediate TextMining with Python

 By: Dr. Eric Godat and Dr. Rob Kalescky

 Adapted from: [Ultimate Guide to deal with Text Data (Using Python)](https://www.analyticsvidhya.com/blog/2018/02/the-different-methods-deal-text-data-predictive-python/)
 
 Natural Language Toolkit: [Documentation](http://www.nltk.org/)
 
 Reference Text: [Natural Language Processing with Python](http://www.nltk.org/book/)


## Setup

These are the basic libraries we will use in for data manipulation (pandas) and math functions (numpy). We will add more libraries as we need them.

As a best practice, it is a good idea to load all your libraries in a single cell at the top of a notebook, however for the purposes of this tutorial we will load some now and more as we go.


```python
%config InlineBackend.figure_format ='retina'
```


```python
import pandas as pd
import numpy as np
import glob
```

Load a data file into a pandas DataFrame.

This tutorial was designed around using sets of data you have yourselves in a form like a CSV, TSV, or TXT file.  Feel free to use any set of data, but for now we will use a dataset created from scraping this [Multilingual Folktale Database](http://www.mftd.org/).

This file is a CSV filetype, common for text data, but your data may also be stored as TSV's, TXT's, or other file types.  This will slightly change how you read from Pandas, but the concept is largely the same for the different filetypes.  Just keep this in mind when you see references to CSV.

To proceed, you will need to have this file downloaded and in the same folder as this notebook. Alternatively you can put the full path to the file.  Typically, your program will look for the file with the name you specified in the folder that contains your program unless you give the program a path to follow.


```python
filename = '../data/folktales.csv'
data = pd.read_csv(filename)
data.rename(columns={'Unnamed: 0':'Index'},inplace=True)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>ATU Code</th>
      <th>Author</th>
      <th>Country of Origin</th>
      <th>Original Title</th>
      <th>Source</th>
      <th>Story</th>
      <th>Story Type</th>
      <th>Title</th>
      <th>Translated</th>
      <th>Year Translated</th>
      <th>Year Written</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>15.0</td>
      <td>Jacob &amp; Wilhelm Grimm</td>
      <td>Germany</td>
      <td>Katze und Maus in Gesellschaft</td>
      <td>NaN</td>
      <td>A cat had made the acquaintance of a mouse, an...</td>
      <td>Stealing the Partner's Butter (ATU 15)\n\t\t</td>
      <td>Cat and mouse in partnership</td>
      <td>Margaret Hunt</td>
      <td>1884.0</td>
      <td>1812</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>123.0</td>
      <td>Jacob &amp; Wilhelm Grimm</td>
      <td>Germany</td>
      <td>Der Wolf und die sieben jungen Geisslein</td>
      <td>NaN</td>
      <td>There was once an old goat who had seven littl...</td>
      <td>The Wolf and the Seven Young Kids (ATU 123)\n...</td>
      <td>The Wolf and the Seven Young Kids</td>
      <td>Margaret Hunt</td>
      <td>1884.0</td>
      <td>1812</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54</td>
      <td>516.0</td>
      <td>Jacob &amp; Wilhelm Grimm</td>
      <td>Germany</td>
      <td>Der treue Johannes</td>
      <td>The Blue Fairy Book (nr. 30)</td>
      <td>ONCE upon a time there was an old king who was...</td>
      <td>The Petrified Friend (ATU 516)\n\t\t</td>
      <td>Trusty John</td>
      <td>Andrew Lang</td>
      <td>1889.0</td>
      <td>1812</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64</td>
      <td>NaN</td>
      <td>Jacob &amp; Wilhelm Grimm</td>
      <td>Germany</td>
      <td>Der gute Handel</td>
      <td>NaN</td>
      <td>There was once a peasant who had driven his co...</td>
      <td>NaN</td>
      <td>The good bargain</td>
      <td>Margaret Hunt</td>
      <td>1884.0</td>
      <td>1812</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>151.0</td>
      <td>Jacob &amp; Wilhelm Grimm</td>
      <td>Germany</td>
      <td>Der wunderliche Spielmann</td>
      <td>NaN</td>
      <td>There was once a wonderful musician, who went ...</td>
      <td>Music lessons for wild animals (ATU 151)\n\t\t</td>
      <td>The Wonderful Musician</td>
      <td>Translated into English</td>
      <td>NaN</td>
      <td>1812</td>
    </tr>
  </tbody>
</table>
</div>




```python
#filename = '/scratch/group/oit_research_data/wikiplots/wikiplots.csv'#If you need to put the path to the file, do so here.
#data = pd.read_csv(filename)
#data.rename(columns={'Unnamed: 0':'Index'},inplace=True)
#data.rename(columns={'Plot':'Story'},inplace=True)
#data = data.sample(2000)
#data.head()
```

Here we can see all the information available to us from the file in the form of a Pandas DataFrame. For the remainder of this tutorial, we will focus primarily on the full text of each data chunk, which we will name the *Story* column.  With your data set this is likely to be something very different, so feel free to call is something else.

## Counting Words and Characters

The first bit of analysis we might want to do is to count the number of words in one piece of data. To do this we will add a column called *wordcount* and write an operation that applies a function to every row of the column.

Unpacking this piece of code, *len(str(x).split(" ")*, tells us what is happening.

For the content of cell *x*, convert it to a string, *str()*, then split that string into pieces at each space, *split()*.

The result of that is a list of all the words in the text and then we can count the length of that list, *len()*.


```python
data['wordcount'] = data['Story'].apply(lambda x: len(str(x).split(" ")))
data[['Story','wordcount']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Story</th>
      <th>wordcount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A cat had made the acquaintance of a mouse, an...</td>
      <td>987</td>
    </tr>
    <tr>
      <th>1</th>
      <td>There was once an old goat who had seven littl...</td>
      <td>1037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ONCE upon a time there was an old king who was...</td>
      <td>3068</td>
    </tr>
    <tr>
      <th>3</th>
      <td>There was once a peasant who had driven his co...</td>
      <td>1723</td>
    </tr>
    <tr>
      <th>4</th>
      <td>There was once a wonderful musician, who went ...</td>
      <td>1066</td>
    </tr>
  </tbody>
</table>
</div>



We can do something similar to count the number of characters in the data chunk, including spaces. If you wanted to exclude whitespaces, you could take the list we made above, join it together and count the length of the resulting string.


```python
data = data.fillna("No Information Provided")
```


```python
data['char_count'] = data['Story'].str.len()
data[['Story','char_count']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Story</th>
      <th>char_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A cat had made the acquaintance of a mouse, an...</td>
      <td>5108</td>
    </tr>
    <tr>
      <th>1</th>
      <td>There was once an old goat who had seven littl...</td>
      <td>5343</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ONCE upon a time there was an old king who was...</td>
      <td>15999</td>
    </tr>
    <tr>
      <th>3</th>
      <td>There was once a peasant who had driven his co...</td>
      <td>8907</td>
    </tr>
    <tr>
      <th>4</th>
      <td>There was once a wonderful musician, who went ...</td>
      <td>5592</td>
    </tr>
  </tbody>
</table>
</div>



Now we want to calculate the average word length in the data.

Let's define a function that will do that for us:


```python
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))
```

We can now apply that function to all the data chunks and save that in a new column.


```python
data['avg_word'] = data['Story'].apply(lambda x: avg_word(x))
data[['Story','avg_word']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Story</th>
      <th>avg_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A cat had made the acquaintance of a mouse, an...</td>
      <td>4.176292</td>
    </tr>
    <tr>
      <th>1</th>
      <td>There was once an old goat who had seven littl...</td>
      <td>4.153327</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ONCE upon a time there was an old king who was...</td>
      <td>4.215124</td>
    </tr>
    <tr>
      <th>3</th>
      <td>There was once a peasant who had driven his co...</td>
      <td>4.170052</td>
    </tr>
    <tr>
      <th>4</th>
      <td>There was once a wonderful musician, who went ...</td>
      <td>4.246717</td>
    </tr>
  </tbody>
</table>
</div>



We can then sort by the average word length.


```python
data[['Story','avg_word']].sort_values(by='avg_word', ascending=True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Story</th>
      <th>avg_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>659</th>
      <td>A GNAT settled on the horn of a Bull, and sa...</td>
      <td>3.623377</td>
    </tr>
    <tr>
      <th>656</th>
      <td>A MAN wished to purchase an Ass, and agreed ...</td>
      <td>3.709924</td>
    </tr>
    <tr>
      <th>620</th>
      <td>A HOUND having started a Hare on the hillsid...</td>
      <td>3.737374</td>
    </tr>
    <tr>
      <th>391</th>
      <td>Once on a time there was a little boy who was ...</td>
      <td>3.771277</td>
    </tr>
    <tr>
      <th>895</th>
      <td>Hare once said to Elephant : " Let us play hid...</td>
      <td>3.795652</td>
    </tr>
  </tbody>
</table>
</div>



# Processing Text

A major component of doing analysis on text is the cleaning of the text prior to the analysis.

Though this process destroys some elements of the text (sentence structure, for example), it is often necessary in order to describe a text analytically. Depending on your choice of cleaning techniques, some elements might be preserved better than others if that is of importance to your analysis.

## Cleaning Up Words

This series of steps aims to clean up and standardize the text itself. This generally consists of removing common elements such as stopwords and punctuation but can be expanded to more detailed removals.

### Lowercase

Here we enforce that all of the text is lowercase. This makes it easier to match cases and sort words.

Notice we are assigning our modified column back to itself. This will save our modifications to our DataFrame


```python
data['Story'] = data['Story'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Story'].head()
```




    0    a cat had made the acquaintance of a mouse, an...
    1    there was once an old goat who had seven littl...
    2    once upon a time there was an old king who was...
    3    there was once a peasant who had driven his co...
    4    there was once a wonderful musician, who went ...
    Name: Story, dtype: object



### Remove Punctuation

Here we remove all punctuation from the data. This allows us to focus on the words only as well as assist in matching.


```python
data['Story'] = data['Story'].str.replace('[^\w\s]','')
data['Story'].head()
```

    /tmp/ipykernel_1623582/3005510997.py:1: FutureWarning: The default value of regex will change from True to False in a future version.
      data['Story'] = data['Story'].str.replace('[^\w\s]','')





    0    a cat had made the acquaintance of a mouse and...
    1    there was once an old goat who had seven littl...
    2    once upon a time there was an old king who was...
    3    there was once a peasant who had driven his co...
    4    there was once a wonderful musician who went q...
    Name: Story, dtype: object



### Remove Stopwords

Stopwords are words that are commonly used and do little to aid in the understanding of the content of a text. There is no universal list of stopwords and they vary on the style, time period and media from which your text came from.  Typically, people choose to remove stopwords from their data, as it adds extra clutter while the words themselves provide little to no insight as to the nature of the data.  For now, we are simply going to count them to get an idea of how many there are.

For this tutorial, we will use the standard list of stopwords provided by the Natural Language Toolkit python library.


```python
#from nltk.corpus import stopwords
#stop = stopwords.words('english')
#data['Story'] = data['Story'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#data['Story'].head()
```

### Remove Frequent Words

If we want to catch common words that might have slipped through the stopword removal, we can build out a list of the most common words remaining in our text.

Here we have built a list of the 10 most common words. Some of these words might actually be relevant to our analysis so it is important to be careful with this method.


```python
#freq = pd.Series(' '.join(data['Story']).split()).value_counts()[:10]
#freq
```

We now follow the same procedure with which we removed stopwords to remove the most frequent words.


```python
#freq = list(freq.index)
#data['Story'] = data['Story'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#data['Story'].head()
```

## Lemmatization

Lemmatization is often a more useful approach than stemming because it leverages an understanding of the word itself to convert the word back to its root word. However, this means lemmatization is less aggressive than stemming (probably a good thing).


```python
#import nltk
#nltk.download('wordnet')
```


```python
from textblob import Word
data['Story'] = data['Story'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Story'].head()
```




    0    a cat had made the acquaintance of a mouse and...
    1    there wa once an old goat who had seven little...
    2    once upon a time there wa an old king who wa s...
    3    there wa once a peasant who had driven his cow...
    4    there wa once a wonderful musician who went qu...
    Name: Story, dtype: object



At this point we have a several options for cleaning and structuring our text data. The next section will focus on more advanced ways to study text analytically.

# Advanced Text Processing

This section focuses on more complex methods of analyzing textual data. We will continue to work with our same DataFrame.

## N-grams

N-grams are combinations of multiple words as they appear in the text. The N refers to the number of words captured in the list. N-grams with N=1 are referred unigrams and are just a nested list of all the words in the text. Following a similar pattern, bigrams (N=2), trigrams (N=3), etc. can be used.

N-grams allow you to capture the structure of the text which can be very useful. For instance, counting the number of bigrams where "said" was preceded by "he" vs "she" could give you an idea of the gender breakdown of speakers in a text. However, if you make your N-grams too long, you lose the ability to make comparisons.

Another concern, especially in very large data sets, is that the memory storage of N-grams scales with N (bigrams are twice as large as unigrams, for example) and the time to process the N-grams can balloon dramatically as well.

All that being said, we would suggest focusing on bigrams and trigrams as useful analysis tools.


```python
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

lemmatizer = WordNetLemmatizer()
n_grams = TextBlob(data['Story'].iloc[1]).ngrams(2)
```


```python
characters=[]
for i in ['he', 'she', 'it', 'they']:
     characters.append(lemmatizer.lemmatize(i))
```


```python
for n in n_grams:
    if n[0] in characters:
        print(n)
```

    ['she', 'had']
    ['she', 'called']
    ['she', 'i']
    ['he', 'were']
    ['he', 'would']
    ['he', 'may']
    ['it', 'wa']
    ['they', 'you']
    ['she', 'ha']
    ['it', 'up']
    ['he', 'came']
    ['he', 'i']
    ['he', 'ran']
    ['he', 'strew']
    ['it', 'cried']
    ['he', 'wa']
    ['he', 'your']
    ['he', 'put']
    ['they', 'saw']
    ['they', 'were']
    ['they', 'opened']
    ['he', 'wa']
    ['they', 'saw']
    ['it', 'wa']
    ['they', 'were']
    ['he', 'swallowed']
    ['he', 'wanted']
    ['he', 'fell']
    ['she', 'sought']
    ['they', 'were']
    ['she', 'called']
    ['she', 'came']
    ['she', 'helped']
    ['she', 'cried']
    ['she', 'wandered']
    ['they', 'came']
    ['they', 'saw']
    ['she', 'noticed']
    ['she', 'can']
    ['it', 'be']
    ['he', 'devoured']
    ['she', 'sent']
    ['she', 'cut']
    ['she', 'made']
    ['they', 'comforted']
    ['he', 'lie']
    ['they', 'fetched']
    ['he', 'wa']
    ['he', 'wa']
    ['they', 'struck']
    ['he', 'cried']
    ['he', 'came']
    ['he', 'fell']
    ['it', 'they']
    ['they', 'came']
    ['they', 'cried']
    ['they', 'danced']



```python
from nltk import ngrams
from collections import Counter

ngram_counts = Counter(ngrams(data['Story'].iloc[1].split(), 2))
for n in ngram_counts.most_common(10):
    print(n)
```

    (('the', 'wolf'), 18)
    (('into', 'the'), 7)
    (('the', 'door'), 7)
    (('but', 'the'), 6)
    (('in', 'the'), 6)
    (('to', 'the'), 6)
    (('and', 'the'), 5)
    (('the', 'mother'), 5)
    (('and', 'when'), 5)
    (('the', 'wood'), 4)


## Term Frequency

Term Frequency is a measure of how often a term appears in a document. There are different ways to define this but the simplest is a raw count of the number of times each term appears.

There are other ways of defining this including a true term frequency and a log scaled definition. All three have been implemented below but the default will the raw count definition, as it matches with the remainder of the definitions in this tutorial.

|Definition|Formula|
|---|---|
|Raw Count|$$f_{t,d}$$|
|Term Frequency|$$\frac{f_{t,d}}{\sum_{t'\in d}f_{t',d}}$$|
|Log Scaled|$$\log(1+f_{t,d})$$|




```python
## Raw Count Definition
tf1 = (data['Story'][0:5]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

## Term Frequency Definition
#tf1 = (data['Story'][0:5]).apply(lambda x: (pd.value_counts(x.split(" ")))/len(x.split(" "))).sum(axis = 0).reset_index() 

## Log Scaled Definition
#tf1 = (data['Story'][0:10]).apply(lambda x: 1.0+np.log(pd.value_counts(x.split(" ")))).sum(axis = 0).reset_index() 

tf1.columns = ['words','tf']
tf1.sort_values(by='tf', ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
      <th>tf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>the</td>
      <td>577.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>and</td>
      <td>354.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>to</td>
      <td>217.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>he</td>
      <td>179.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>157.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>it</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>of</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>i</td>
      <td>107.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>his</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>that</td>
      <td>96.0</td>
    </tr>
  </tbody>
</table>
</div>



## Inverse Document Frequency

Inverse Document Frequency is a measure of how common or rare a term is across multiple documents. That gives a measure of how much weight that term carries.

For a more concrete analogy of this, imagine a room full of NBA players; here a 7 foot tall person wouldn't be all that shocking. However if you have a room full of kindergarten students, a 7 foot tall person would be a huge surprise.

The simplest and standard definition of Inverse Document Frequency is to take the logarithm of the ratio of the number of documents containing a term to the total number of documents.

$$-\log\frac{n_t}{N} = \log\frac{N}{n_t}$$



```python
for i,word in enumerate(tf1['words']):
    tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Story'].str.contains(word)])))

tf1[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
      <th>tf</th>
      <th>idf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>the</td>
      <td>577.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>and</td>
      <td>354.0</td>
      <td>0.013304</td>
    </tr>
    <tr>
      <th>2</th>
      <td>to</td>
      <td>217.0</td>
      <td>0.009961</td>
    </tr>
    <tr>
      <th>3</th>
      <td>it</td>
      <td>124.0</td>
      <td>0.034737</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>157.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>you</td>
      <td>84.0</td>
      <td>0.163556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>of</td>
      <td>111.0</td>
      <td>0.041602</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cat</td>
      <td>21.0</td>
      <td>1.162270</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mouse</td>
      <td>20.0</td>
      <td>2.591737</td>
    </tr>
    <tr>
      <th>9</th>
      <td>said</td>
      <td>85.0</td>
      <td>0.255887</td>
    </tr>
  </tbody>
</table>
</div>



## Term Frequency – Inverse Document Frequency ([TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf))

Term Frequency – Inverse Document Frequency (TF-IDF) is a composite measure of both Term Frequency and Inverse Document Frequency.

From [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf):
"A high weight in TF–IDF is reached by a high term frequency (in the given document) and a low document frequency of the term in the whole collection of documents; the weights hence tend to filter out common terms"

More concisely, a high TD-IDF says that a word is very important in the documents in which it appears.

There are a few weighting schemes for TF-IDF. Here we use scheme (1).

|Weighting Scheme|Document Term Weight|
|---|---|
|(1)|$$f_{t,d}\cdot\log\frac{N}{n_t}$$|
|(2)|$$1+\log(f_{t,d})$$|
|(3)|$$(1+\log(f_{t,d}))\cdot\log\frac{N}{n_t}$$|


```python
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1.sort_values(by='tfidf', ascending=False)[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>words</th>
      <th>tf</th>
      <th>idf</th>
      <th>tfidf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>541</th>
      <td>john</td>
      <td>39.0</td>
      <td>4.038656</td>
      <td>157.507571</td>
    </tr>
    <tr>
      <th>542</th>
      <td>trusty</td>
      <td>37.0</td>
      <td>4.172187</td>
      <td>154.370921</td>
    </tr>
    <tr>
      <th>987</th>
      <td>aik</td>
      <td>20.0</td>
      <td>6.811244</td>
      <td>136.224888</td>
    </tr>
    <tr>
      <th>1183</th>
      <td>musician</td>
      <td>31.0</td>
      <td>4.038656</td>
      <td>125.198325</td>
    </tr>
    <tr>
      <th>985</th>
      <td>peasant</td>
      <td>25.0</td>
      <td>2.548565</td>
      <td>63.714113</td>
    </tr>
    <tr>
      <th>342</th>
      <td>wolf</td>
      <td>27.0</td>
      <td>2.278645</td>
      <td>61.523412</td>
    </tr>
    <tr>
      <th>991</th>
      <td>wow</td>
      <td>12.0</td>
      <td>4.731803</td>
      <td>56.781634</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mouse</td>
      <td>20.0</td>
      <td>2.591737</td>
      <td>51.834733</td>
    </tr>
    <tr>
      <th>988</th>
      <td>jew</td>
      <td>16.0</td>
      <td>2.750801</td>
      <td>44.012822</td>
    </tr>
    <tr>
      <th>344</th>
      <td>kid</td>
      <td>10.0</td>
      <td>3.920873</td>
      <td>39.208726</td>
    </tr>
  </tbody>
</table>
</div>



It is worth noting that the *sklearn* library has the ability to directly calculate a TD-IDF matrix.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
data_vect = tfidf.fit_transform(data['Story'])

data_vect
```




    <908x1000 sparse matrix of type '<class 'numpy.float64'>'
    	with 122671 stored elements in Compressed Sparse Row format>



## Similarity


```python
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # This suppresses a warning from scikit learn that they are going to update their code
```

One thing we can look at is how similar two texts are. This has a practical use when looking for plagiarism, but can also be used to compare author's styles. To do this there are a few ways we can measure similarity.

First we need to set up our two sets of texts. Here we have the ability to choose the size of our sets and whether we want the first n texts from our full data set or just a random sample. Keep in mind that if you want to use two dissimilar sets, you won't have a control value (something x itself).


```python
size=10
```


```python
## First n by First n
#set1 = data[["Index",'Story','Title']][:size]
#set2 = data[["Index",'Story','Title']][:size]

## Random X Itself
set1 = data[["Index",'Story','Title']].sample(size)
set2 = set1

## First n by Random
#set1 = data[["Index",'Story','Title']][:size]
#set2 = data[["Index",'Story','Title']].sample(size)

## Random X Random
#set1 = data[["Index",'Story','Title']].sample(size)
#set2 = data[["Index",'Story','Title']].sample(size)
```


```python
#To let us do a "cross join": Every row in one gets matched to all rows in the other
set1['key']=1 
set2['key']=1

similarity = pd.merge(set1, set2, on ='key', suffixes=('_1', '_2')).drop("key", 1)
```

### Jaccard Similarity

The first is using a metric called the [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index). This is just taking the intersection of two sets of things (in our case, words or n-grams) and dividing it by the union of those sets. This gives us a metric for understanding how the word usage compares but doesn't account for repeated words since the union and intersections just take unique words. One advantage though is that we can easily extend the single word similarity to compare bi-grams and other n-grams if we want to examine phrase usage.

$$S_{J}(A,B)=\frac{A \cap B}{A \cup B}$$



```python
def jaccard(row, n=1):
    
    old=row['Story_1']
    new=row['Story_2']
    
    old_n_grams = [tuple(el) for el in TextBlob(old).ngrams(n)]
    new_n_grams = [tuple(el) for el in TextBlob(new).ngrams(n)]
        
    union = list(set(old_n_grams) | set(new_n_grams))
    intersection = list(set(old_n_grams) & set(new_n_grams))
    
    lu = len(union)
    li = len(intersection)
    
    return (li/lu,li,lu)
```


```python
for i in [1,2]: # Add values to the list for the n-gram you're interested in
    similarity['Jaccard_Index_for_{}_grams'.format(i)]=similarity.apply(lambda x: jaccard(x,i)[0],axis=1)
```


```python
similarity['Intersection']=similarity.apply(lambda x: jaccard(x)[1],axis=1)
```


```python
similarity['Union']=similarity.apply(lambda x: jaccard(x)[2],axis=1)
```


```python
similarity.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index_1</th>
      <th>Story_1</th>
      <th>Title_1</th>
      <th>Index_2</th>
      <th>Story_2</th>
      <th>Title_2</th>
      <th>Jaccard_Index_for_1_grams</th>
      <th>Jaccard_Index_for_2_grams</th>
      <th>Intersection</th>
      <th>Union</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>341</td>
      <td>341</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>5496</td>
      <td>a fox wa mounting a hedge when he lost his foo...</td>
      <td>The Fox and the Bramble</td>
      <td>0.089674</td>
      <td>0.009060</td>
      <td>33</td>
      <td>368</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>5335</td>
      <td>a bat who fell upon the ground and wa caught b...</td>
      <td>The Bat and the Weasels</td>
      <td>0.078591</td>
      <td>0.014590</td>
      <td>29</td>
      <td>369</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>2257</td>
      <td>yes they called him little tuk but it wa not h...</td>
      <td>Little Tuk</td>
      <td>0.211566</td>
      <td>0.041667</td>
      <td>150</td>
      <td>709</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>2725</td>
      <td>the wind whistle in the old willow tree it is ...</td>
      <td>What old Johanne told</td>
      <td>0.163820</td>
      <td>0.036155</td>
      <td>211</td>
      <td>1288</td>
    </tr>
  </tbody>
</table>
</div>



### Cosine Similarity

The second metric we can use is [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity), however there is a catch here. Cosine similarity requires a vector for each word so we make a choice here to use term frequency. You could choose something else, inverse document frequency or tf-idf would both be good choices. Cosine similarity with a term frequency vector gives us something very similar to the Jaccard Index but accounts for word repetition. This makes it better for tracking word importance between two texts.

$$S_{C}(v_1,v_2)=cos(\theta)=\frac{v_1\cdot v_2}{||v_1||\times||v_2||}$$


```python
def get_cosine_sim(str1,str2): 
    vectors = [t for t in get_vectors([str1,str2])]
    return cosine_similarity(vectors)
    
def get_vectors(slist):
    text = [t for t in slist]
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def cosine(row):
    
    old=row['Story_1']
    new=row['Story_2']
    
    return get_cosine_sim(old,new)[0,1]   
```


```python
similarity['Cosine_Similarity']=similarity.apply(lambda x: cosine(x),axis=1)
```


```python
similarity.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index_1</th>
      <th>Story_1</th>
      <th>Title_1</th>
      <th>Index_2</th>
      <th>Story_2</th>
      <th>Title_2</th>
      <th>Jaccard_Index_for_1_grams</th>
      <th>Jaccard_Index_for_2_grams</th>
      <th>Intersection</th>
      <th>Union</th>
      <th>Cosine_Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>341</td>
      <td>341</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>5496</td>
      <td>a fox wa mounting a hedge when he lost his foo...</td>
      <td>The Fox and the Bramble</td>
      <td>0.089674</td>
      <td>0.009060</td>
      <td>33</td>
      <td>368</td>
      <td>0.544462</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>5335</td>
      <td>a bat who fell upon the ground and wa caught b...</td>
      <td>The Bat and the Weasels</td>
      <td>0.078591</td>
      <td>0.014590</td>
      <td>29</td>
      <td>369</td>
      <td>0.627578</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>2257</td>
      <td>yes they called him little tuk but it wa not h...</td>
      <td>Little Tuk</td>
      <td>0.211566</td>
      <td>0.041667</td>
      <td>150</td>
      <td>709</td>
      <td>0.795735</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>Cat and mouse in partnership</td>
      <td>2725</td>
      <td>the wind whistle in the old willow tree it is ...</td>
      <td>What old Johanne told</td>
      <td>0.163820</td>
      <td>0.036155</td>
      <td>211</td>
      <td>1288</td>
      <td>0.841591</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize Similarity


```python
metric = 'Cosine_Similarity'
#metric = 'Jaccard_Index_for_2_grams'
```


```python
import numpy as np 
import matplotlib.pyplot as plt

index = list(similarity['Index_1'].unique())
columns = list(similarity['Index_2'].unique())
df = pd.DataFrame(0, index=index, columns=columns)

for i in df.index:
    sub = similarity[(similarity['Index_1']==i)]
    for col in df.columns:
        df.loc[i,col]=sub[sub['Index_2']==col][metric].iloc[0]
        
plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.show()
```


    
![png](output_85_0.png)
    



```python
similarity[['Title_1','Index_1','Title_2','Index_2',metric]].sort_values(by=metric,ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title_1</th>
      <th>Index_1</th>
      <th>Title_2</th>
      <th>Index_2</th>
      <th>Cosine_Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>Baba-Yaga and Puny</td>
      <td>3024</td>
      <td>Baba-Yaga and Puny</td>
      <td>3024</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>88</th>
      <td>Poverty and humility lead to heaven</td>
      <td>1922</td>
      <td>Poverty and humility lead to heaven</td>
      <td>1922</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The Fox and the Bramble</td>
      <td>5496</td>
      <td>The Fox and the Bramble</td>
      <td>5496</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>The Panther and Shepherds</td>
      <td>7984</td>
      <td>The Fox and the Bramble</td>
      <td>5496</td>
      <td>0.413735</td>
    </tr>
    <tr>
      <th>25</th>
      <td>The Bat and the Weasels</td>
      <td>5335</td>
      <td>The Ass and the Lion Hunting</td>
      <td>7948</td>
      <td>0.413626</td>
    </tr>
    <tr>
      <th>52</th>
      <td>The Ass and the Lion Hunting</td>
      <td>7948</td>
      <td>The Bat and the Weasels</td>
      <td>5335</td>
      <td>0.413626</td>
    </tr>
    <tr>
      <th>51</th>
      <td>The Ass and the Lion Hunting</td>
      <td>7948</td>
      <td>The Fox and the Bramble</td>
      <td>5496</td>
      <td>0.399667</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The Fox and the Bramble</td>
      <td>5496</td>
      <td>The Ass and the Lion Hunting</td>
      <td>7948</td>
      <td>0.399667</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 5 columns</p>
</div>




```python
similarity[similarity[metric]<0.999][['Title_1','Index_1','Title_2','Index_2',metric]].sort_values(by=metric,ascending=False)[:12]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title_1</th>
      <th>Index_1</th>
      <th>Title_2</th>
      <th>Index_2</th>
      <th>Cosine_Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>What old Johanne told</td>
      <td>2725</td>
      <td>0.909880</td>
    </tr>
    <tr>
      <th>43</th>
      <td>What old Johanne told</td>
      <td>2725</td>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>0.909880</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>What old Johanne told</td>
      <td>2725</td>
      <td>0.896002</td>
    </tr>
    <tr>
      <th>46</th>
      <td>What old Johanne told</td>
      <td>2725</td>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>0.896002</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>Baba-Yaga and Puny</td>
      <td>3024</td>
      <td>0.892949</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Baba-Yaga and Puny</td>
      <td>3024</td>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>0.892949</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Poverty and humility lead to heaven</td>
      <td>1922</td>
      <td>What old Johanne told</td>
      <td>2725</td>
      <td>0.879828</td>
    </tr>
    <tr>
      <th>48</th>
      <td>What old Johanne told</td>
      <td>2725</td>
      <td>Poverty and humility lead to heaven</td>
      <td>1922</td>
      <td>0.879828</td>
    </tr>
    <tr>
      <th>63</th>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>0.871639</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>Farmer Weathersky</td>
      <td>3283</td>
      <td>0.871639</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Baba-Yaga and Puny</td>
      <td>3024</td>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>0.870745</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Little Tuk</td>
      <td>2257</td>
      <td>Baba-Yaga and Puny</td>
      <td>3024</td>
      <td>0.870745</td>
    </tr>
  </tbody>
</table>
</div>



## [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model)

[Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) a way to represent text based on the idea that similar texts will contain similar vocabulary. There is a lot to this model and we provide merely a simple implementation of it here.


```python
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(data['Story'])
data_bow
```




    <908x1000 sparse matrix of type '<class 'numpy.int64'>'
    	with 189392 stored elements in Compressed Sparse Row format>



## Sentiment Analysis

Sentiment is a way of measuring the overall positivity or negativity in a given text.

To do this we will use the built in sentiment function in the *TextBlob* package. This function will return the polarity and subjectivity scores for each data chunk.


```python
data['Story'][:5].apply(lambda x: TextBlob(x).sentiment)
```




    0     (0.1942645317645318, 0.48949938949938954)
    1    (-0.03807288173485357, 0.3989380728817349)
    2     (0.21112742728224962, 0.5152775763435665)
    3     (0.10027879728966686, 0.5330980613589311)
    4     (0.06213682941531042, 0.4784448462929475)
    Name: Story, dtype: object



Focusing on the polarity score, we are able to see the overall sentiment of each data chunk. The closer to 1 the more positive and the closer to -1 the more negative.


```python
data['sentiment'] = data['Story'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['Story','sentiment']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Story</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a cat had made the acquaintance of a mouse and...</td>
      <td>0.194265</td>
    </tr>
    <tr>
      <th>1</th>
      <td>there wa once an old goat who had seven little...</td>
      <td>-0.038073</td>
    </tr>
    <tr>
      <th>2</th>
      <td>once upon a time there wa an old king who wa s...</td>
      <td>0.211127</td>
    </tr>
    <tr>
      <th>3</th>
      <td>there wa once a peasant who had driven his cow...</td>
      <td>0.100279</td>
    </tr>
    <tr>
      <th>4</th>
      <td>there wa once a wonderful musician who went qu...</td>
      <td>0.062137</td>
    </tr>
  </tbody>
</table>
</div>



Here we have textted the sentiment scores for the first 10 chunks.

Notice they tend to be positive but not exceedingly so.


```python
plot_1 = data[['sentiment']][:10].plot(kind='bar')
```


    
![png](output_97_0.png)
    


Now we have sorted and textted all of the sentiment scores for the chunks in our database.

We can clearly see that most of the text data is positive but not overwhelmingly so (as seen by the long tail of the distribution). However, the parts that are negative tend to be more polarized than the positive ones (a shorter tail and sharper peak).


```python
plot_2 = data[['sentiment']].sort_values(by='sentiment', ascending=False).plot(kind='bar')
```


    
![png](output_99_0.png)
    


## Using TF-IDF and Machine Learning

This is significantly more advanced than the rest of the tutorial. This takes the TF-IDF matrix and applies a [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering). This groups the texts into clusters of similar terms from the TF-IDF matrix. This algorithm randomly seeds X "means", the values are then clustered into the nearest mean. The centroid of the values in each cluster then becomes the new mean and the process repeats until a convergence is reached.


```python
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

groups = 15

num_clusters = groups
num_seeds = groups
max_iterations = 300

cmap = matplotlib.cm.get_cmap("nipy_spectral", groups) # Builds a discrete color mapping using a built in matplotlib color map

c = {}
for i in range(cmap.N): # Converts our discrete map into Hex Values
    rgba = cmap(i)
    c.update({i:matplotlib.colors.rgb2hex(rgba)})

labels_color_map=c

pca_num_components = 2
tsne_num_components = 2

# calculate tf-idf of texts
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
tf_idf_matrix = tfidf.fit_transform(data['Story'])

# create k-means model with custom config
clustering_model = KMeans(
    n_clusters=num_clusters,
    max_iter=max_iterations,
    #precompute_distances="auto",
    #n_jobs=-1
)

labels = clustering_model.fit_predict(tf_idf_matrix)
#print(labels)

X = tf_idf_matrix.todense()

# ----------------------------------------------------------------------------------------------------------------------

reduced_data = PCA(n_components=pca_num_components).fit_transform(X)
# print(reduced_data)

import matplotlib.patches as mpatches
legendlist=[mpatches.Patch(color=labels_color_map[key],label=str(key))for key in labels_color_map.keys()]

fig, ax = plt.subplots()
for index, instance in enumerate(reduced_data):
    #print(instance, index, labels[index])
    pca_comp_1, pca_comp_2 = reduced_data[index]
    color = labels_color_map[labels[index]]
    ax.scatter(pca_comp_1, pca_comp_2, c=color)
plt.legend(handles=legendlist)
plt.show()



# t-SNE plot
#embeddings = TSNE(n_components=tsne_num_components)
#Y = embeddings.fit_transform(X)
#plt.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
#plt.show()

```


    
![png](output_102_0.png)
    



```python
tfidf_test = tf1.sort_values(by='tfidf', ascending=False)[:1000]
```


```python
data['Story Type'] = data['Story Type'].str.strip()
```


```python
title_groups = np.transpose([labels,data['Title'],data['Story Type']])
```


```python
#title_groups = np.transpose([labels,data['Title']])
```

These are the titles of the texts in each cluster. Keep in mind that each time you run the algorithm, the randomness in generating the initial means will result in different clusters.


```python
for i in range(len(title_groups)):
    data.loc[i,'Group'] = title_groups[i][0]
```


```python
grp=0
data[data['Group']==grp]['Title']#.sample(15)
```




    166                                The wicked prince
    299                                  Right and Wrong
    300                  Prince Ivan and Princess Martha
    301                               The Three Kingdoms
    304                                    Marya Morevna
    305    King Ivan and Bely, the Warrior of the Plains
    310      The King of the Sea and Vassilissa the Wise
    313                              The Prophetic Dream
    324                     Tsarevich Ivan and Grey Wolf
    467                                The Princess Frog
    865                                 Gesture of Death
    905                           The gardiner and Death
    Name: Title, dtype: object




```python
for i in range(groups):
    print("")
    print("#### {} ###".format(i))
    print("")
    for el in title_groups:
        if el[0]==i:
            print("{}".format(el[1]))
```

    
    #### 0 ###
    
    The wicked prince
    Right and Wrong
    Prince Ivan and Princess Martha
    The Three Kingdoms
    Marya Morevna
    King Ivan and Bely, the Warrior of the Plains
    The King of the Sea and Vassilissa the Wise
    The Prophetic Dream
    Tsarevich Ivan and Grey Wolf
    The Princess Frog
    Gesture of Death
    The gardiner and Death
    
    #### 1 ###
    
    Herr Korbes
    The death of the little hen
    The golden key
    There is no doubt about it
    The farm-yard cock and the weather-cock
    The gate key
    The Cock and Hen
    The Box With Something Pretty In It
    Blue Beard
    The Frog and Ox
    The Hen and the Golden Eggs
    The Woman and Her Hen
    The Hen and the Swallow
    The Dog and the Oyster
    The Cobbler and His Three Daughters
    The Goose that Laid the Golden Egg
    
    #### 2 ###
    
    Straw, coal, and bean
    The Valiant Little Tailor
    The Dog and the Sparrow
    Clever Grethel
    Old Hildebrand
    Sweet Porridge
    Fair Katrinelje and Pif-Paf-Poltrie
    Knoist and his three sons
    Simeli mountain
    The ungrateful son
    The tale of Cockaigne
    The peasant in heaven
    The nail
    The hare and the hedgehog
    The crumbs on the table
    Little Claus and big Claus
    A picture from the ramparts
    In a thousand years
    The swan's nest
    Two maidens
    The thorny road of honor
    The bell deep
    The races
    The rags
    The candles
    Dance, dance, doll of mine!
    The Tallow Candle
    Emelya and the Pike
    The Horse, the Table-Cloth and the Horn
    The Mountain of Gold
    Ivan the Fool
    Good But Bad
    The Miser
    One's Own Children Are Always Prettiest
    The Cock, the Cuckoo, and the Blackcock
    The Three Billy-Goats Gruff
    The Cock and Hen a-Nutting
    Goodman Axehaft
    The Hare And The Heiress
    The Knights of the Fish
    The Cat and the Fox
    The Two Mules
    The Tartaro
    The Tortoise and the Eagle
    The Baker's Apprentice
    Don Firriulieddu
    The Ass and the Grasshopper 
    The Charcoal-Burner and the Fuller
    The Boy Hunting Locusts
    The Ants and the Grasshopper
    The Dog and the Shadow
    The Mole and His Mother
    The Pomegranate, Apple-Tree, and Bramble
    The Farmer and the Stork
    The Fawn and His Mother
    The Swallow and the Crow
    The Flies and the Honey-Pot
    The Bear and the Two Travelers
    The Thirsty Pigeon
    The Miser
    The Horse and Groom
    The Ass and the Lapdog
    The Ass and the Mule
    The Salt Merchant and His Ass
    The Vain Jackdaw
    The Mischievous Dog
    The Charger and the Miller
    The Horse and His Rider
    The Belly and the Members
    The Vine and the Goat
    The Widow and Her Little Maidens
    The Father and His Two Daughters
    The Farmer and His Sons
    The Crab and Its Mother
    The Swallow, the Serpent, and the Court of Justice
    The Thief and His Mother
    The Fir-Tree and the Bramble
    The Man Bitten by a Dog
    The Two Pots
    The Aethiop
    The Two Dogs
    The Eagle and the Arrow
    The Sick Kite
    The One-Eyed Doe
    The Ass Carrying the Image
    The Bee and Jupiter
    The Seaside Travelers
    The Brazier and His Dog
    The Ass and His Masters
    The Oak and the Reeds
    Mercury and the Sculptor
    The Flea and the Wrestler
    The Peasant and the Eagle
    The Image of Mercury and the Carpenter
    The Bald Knight
    The Lamp
    The Crow and the Raven
    The Kites and the Swans
    The Camel
    The Wasp and the Snake
    The Dog and the Hare
    The Peacock and the Crane
    The Mule
    The Hart and the Vine
    The Walnut-Tree
    The Jackdaw and the Doves
    The Horse and the Stag
    The Prophet
    The Man and His Wife
    The Fowler and the Viper
    The Geese and the Cranes
    The North Wind and the Sun
    The Two Men Who Were Enemies
    The Gamecocks and the Partridge
    The Spendthrift and the Swallow
    The Dove and the Crow
    The Man and the Satyr
    The Ass and His Purchaser
    The Dogs and the Hides
    The Monkey and the Camel
    Truth and the Traveler
    The Crow and the Serpent
    The Ant and the Dove
    The Thieves and the Cock
    The Dog and the Cook
    The Travelers and the Plane-Tree
    The Rich Man and the Tanner
    The Shipwrecked Man and the Sea
    The Ass and the Charger
    The Ass and His Driver
    The Thrush and the Fowler
    The Rose and the Amaranth
    Thirteenth
    The Steward of Skalholt.
    The Mirror of Truth
    The Two Travellers
    The Inquisitive Cat
    The Carp and her Young
    The Two Gardeners
    The Stag at the Fountain
    The Bitch and Her Puppies
    The Faithful House-dog
    The Dog, Treasure, and Vulture
    The Kite and the Doves
    The Dog in the River
    The Man and the Dog
    Of Doubt and Credulity
    The Cock and the Pearl
    The Owl and the Grasshopper
    The Trees Protected
    Juno and the Peacock
    Esop and the Importunate Fellow
    The Ape's Head
    Esop and the Insolent Fellow
    The Ass and Priests of Cybele
    The Sacrilegious Thief
    Hercules and Plutus
    The He-goats and She-goats
    The Man and the Adder
    The Escape of Simonides
    The Horse and Boar
    The Viper and the File
    Demetrius and Menander
    The Old Dog and the Huntsman
    The Man and the Ass
    The Two Bald Men
    Opportunity
    The Man and the Dog
    A Saying of Socrates
    The Swan, The Pike and the Crayfish 
    The Ant
    Elephant And Pug
    The ant and the Grasshopper
    A Hog under an Oak
    Dragonfly and the Ant
    The Grasshopper and the Ant
    The Rat and the Elephant
    The Hare and the Partridge
    The Gardener and his Landlord
    The Rat Retired From the World
    The Wishes
    The Dairy-woman and the Pail of Milk
    The Man Who Ran After Fortune and The Man Who Waited For her In his Bed
    The Power of Fable
    The Dog Who Carried his Master's Dinner
    The Horoscope
    Education
    The Acorn and the Pumpkin
    The Schoolboy, the Pedant, and the Owner of a Garden
    The Sculptor and the Statue of Jupiter
    The Oyster and the Pleaders
    The Two Rats, the Fox, and the Egg
    The Dog With his Ears Cropped
    The Rabbits
    The Gods Wishing To Instruct a Son of Jupiter
    Love and Folly
    The Forest and the Woodcutter
    The Ape
    The Scythian Philosopher
    The Arbiter, the Hospitaller, and the Hermit
    The Man and the Adder
    The Rooster and the Gem
    The Eagle and the Jackdaw
    The Fly and the Bee
    The Crow and the Peacock
    The Man and the Viper
    The Hawk and the Nightingale
    The Turtoise and the Geese
    When Death Came to Baghdad
    The Linguist and the Dervish
    The Rooster and the Hyena
    The Altercation between Spring and Summer 
    The Friendship of the Thorn and the Mud-Fish 
    The Black Monkey and the Turtoise
    Copa Nanzi and the Cow
    The Town Rat and the Country Rat
    How Hare Scared Elephant 
    The young man and the monitor lizard
    
    #### 3 ###
    
    Brides on their trial
    Choosing a Bride
    The Fool and the Birch Tree
    The Oxen and the Axle-Trees
    The Piglet, the Sheep, and the Goat
    The Heifer and the Ox
    The Widow and the Sheep
    The Shepherd and the Sea
    The Master and His Dogs
    The Two Travelers and the Axe
    The Oaks and Jupiter
    The Traveler and Fortune
    The Goat and the Ass
    The Wasps, the Partridges, and the Farmer
    The Flea and the Ox
    Mercury and the Workmen
    The Shepherd and the Sheep
    The Peasant and the Apple-Tree
    The Partridge and the Fowler
    The Flea and the Man
    The Hungry Dogs
    The Proud Frog
    The Dog and the Crocodile
    The Bees and the Drone
    The Dog and the Lamb
    The Panther and Shepherds
    The Fly and the Mule
    The Stag and the Oxen
    The Animals Sick of the Plague
    Thyrsis and Amaranth
    The Dog and the Cheese
    The Appointment in Samarra
    The Ass and the Flute
    The Two Flies and the Bee
    The Blue Lily
    
    #### 4 ###
    
    The good bargain
    The twelve brothers
    The girl without hands
    The three languages
    The tailor in heaven
    The Wishing-Table, the Gold-Ass, and the Cudgel in the Sack
    Frau Trude
    Bird-foundling
    The knapsack, the hat, and the horn
    The little peasant
    Brother Lustig
    The singing, springing lark
    The young giant
    The king of the golden mountain
    The raven
    Bearskin
    The two travellers
    The blue light
    The old woman in the wood
    Ferdinand the faithful
    The iron stove
    One-eye, two-eyes, and three-eyes
    The six servants
    Iron John
    The three black princesses
    Domestic servants
    Going a-travelling
    The donkey
    The turnip
    The sparrow and his four children
    A riddling tale
    Lazy Harry
    Lean Lisa
    The duration of life
    The goose-girl at the well
    The nix of the mill-pond
    The giant and the tailor
    The poor boy in the grave
    The true bride
    The peasant and the devil
    The sea-hare
    The master-thief
    The drummer
    The grave-mound
    The crystal ball
    Maid Maleen
    The boots of buffalo-leather
    The three green twigs
    The Phoenix bird
    The Sevens Bits of Bacon-Rind
    The Young Slave
    The Frog-King, or Iron Henry
    Our Lady's Child
    The Story of the Youth who Went Forth to Learn What Fear Was
    The Crow and Mercury
    Esop Playing
    The Bald Man and the Fly
    The Bull and the Calf
    Maol a chliobain
    Asphurtzela
    
    #### 5 ###
    
    The Kingdom of the Lion
    The Herdsman and the Lost Bull
    The Lioness
    The Wild Ass and the Lion
    The Lion and the Dolphin
    The Lion and the Boar
    The Ass, the Cock, and the Lion
    The Lion in a Farmyard
    The Doe and the Lion
    The Lion and the Hare
    The Bull and the Goat
    The Lion, the Fox, and the Ass
    The Gnat and the Lion
    The Lion and the Bull
    The Wolf and the Lion
    The Manslayer
    The Lion, Jupiter, and the Elephant
    The Viper and the File
    The King's Son and the Painted Lion
    The Ass and the Lion Hunting
    The Old Lion
    The Judicious Lion
    The Lion and the Mouse
    The Lion and the Three Bulls
    The Lion and the Shepherd
    The Lion, the Panther, the Tazourit, and the Jackal
    The Cow, the Goat, the Sheep, and the Lion
    
    #### 6 ###
    
    The Animals' Winter Home
    The Tale of Ruff Ruffson, Son of Bristle
    Two bulls and the frog
    The Frog and the Ox
    The Fisherman Piping
    The Fisherman and His Nets
    The Fisherman and the Little Fish
    The Monkey and the Fishermen
    The Two Frogs
    The Seagull and the Kite
    The Ass and the Frogs
    The Two Frogs
    The Fishermen
    The Two Bags
    The Gnat and the Bull
    The Hares and the Frogs
    The Camel and Jupiter
    The Frogs' Complaint Against the Sun
    The Frog and the Rat
    The Frogs Desiring a King
    The Frogs and Sun
    The Two Bags
    The Frogs and the Fighting Bulls
    The Bull, the Tup, the Cock, and the Steg
    
    #### 7 ###
    
    Trusty John
    Rapunzel
    The three snake-leaves
    The white snake
    Cinderella
    The riddle
    The singing bone
    The devil with the three golden hairs
    The six swans
    Sleeping Beauty (Little Briar Rose)
    King Thrushbeard
    Rumpelstiltskin
    The two brothers
    The queen bee
    The three feathers
    The golden goose
    All-kinds-of-fur
    The twelve huntsmen
    The three children of fortune
    Six soldiers of fortune
    The goose girl
    The gnome
    The three little birds
    The water of life
    Hans my Hedgehog
    The skilful huntsman
    The four skilful brothers
    The shoes that were danced to pieces
    The white bride and the black one
    Old Rinkrank
    Poverty and humility lead to heaven
    The princess and the pea
    The travelling companion
    The wild swans
    The swineherd
    The jumper
    The White Duck
    Evening, Midnight and Dawn
    The Fire-Bird and Princess Vassilissa
    The Riddle
    True and Untrue
    Hacon Grizzlebeard
    The Twelve Wild Ducks
    Shortshanks
    Tatterhood
    Katie Woodencloak
    The Big Bird Dan
    Little Annie the Goose-Girl
    The Green Knight
    The Golden Bird
    King Valemon, The White Bear
    The Three Lemons
    The Priest And The Clerk
    The Charcoal-burner
    Peik
    Rumble-Mumble Goose-Egg
    The Sleeping Beauty in the Wood
    The Myrtle
    Cenerentola
    The Master Cat; or, Puss in Boots
    Cinderella; or, The Little Glass Slipper
    Ricky of the Tuft
    Sun, Moon, and Talia
    Peruonto
    The Flea
    The Merchant
    Goat-Face
    The Enchanted Doe
    The Three Sisters
    Violet
    Pippo
    The She-Bear
    The Dove
    Canntella
    Corvetto
    The Booby
    The Stone in the Cock's Head
    The Three Enchanted Princes
    The Dragon
    The Two Cakes
    The Raven
    Pintosmalto
    Nennillo and Nennella
    The Three Citrons
    The Morning Star And The Evening Star
    Youth Without Age And Life Without Death
    Baldak Borisevich
    The Magic Ring
    The Golden Horseshoe the Golden Feather and the Golden Hair
    Godmother Death and the Doctor Who Worked Miracles
    Three Coppers
    The Bird of Truth
    The Princess Bella-Flor
    The Blue Bird
    Tale of the dead princessand the seven knights
    Anansi in hunger
    The Hearth-Cat
    The Fire-blower
    Tartaro and Petit Perroquet
    Faithful John
    The Stepmother
    The King of Love
    The dancing water, the singing apple, and the speaking bird
    The Cistern
    Snow-White-Fire-Red
    Water and Salt
    The Cottager and his Cat
    Litill, Tritill, the Birds and the Peasant Lad
    Asmund and Signy
    Geirlug the King's Daughter
    The Cobbler Turned Doctor
    The Witch in the Stone Boat
    Bash-Chalek; Or, True Steel
    The Lion, the Monkey, and the Two Asses
    Prince Hyacinth and the Dear Little Princess
    East of the Sun and West of the Moon
    The Yellow Dwarf
    Rumpelstiltzkin
    The Master-Maid
    The White Cat
    Felicia and the Pot of Pinks
    The Wonderful Sheep
    The Story of Pretty Goldilocks
    The Goose-Girl
    Toads and Diamonds
    The Princess on the Glass Hill
    Princess Rosette
    The Knights of Fish
    The Little Good Mouse
    The Golden Branch
    
    #### 8 ###
    
    The Wonderful Musician
    The golden bird
    The wolf and the fox
    Gossip wolf and the fox
    The fox and the cat
    The fox and the geese
    The fox and the horse
    The Cat, the Rooster and the Fox
    The Animals in the Pit
    The Fox and the Crane
    The Fox as Herdsman
    Bear and Bruin
    Well Done and Ill Paid
    The Cock and Hen That Went to the Dovrefell
    Nanny Who Wouldn't Go Home to Supper
    Snegurushka and the Fox
    Liza the Fox and Catafay the Cat
    Fox, Hare and Cock
    The Crow and the Fox
    The Eagle and the Fox 
    The Fox and the Crow
    The Fox and the Grapes
    The Bear and the Fox
    The Ass, the Fox, and the Lion
    The Fox and the Goat
    The Fox Who Had Lost His Tail
    The Wild Boar and the Fox
    The Fox and the Woodcutter
    The Lion, the Bear, and the Fox
    The Fox and the Leopard
    The Crab and the Fox
    The Hares and the Foxes
    The Dog, the Cock, and the Fox
    The Lion, the Wolf, and the Fox
    The Jackdaw and the Fox
    The Fox and the Bramble
    The Ass in the Lion's Skin
    The Wonderful Musician
    The Wolf and Fox, with the Ape for Judge
    The Fox and the Crow
    The Fox and the Stork
    The Fox and Eagle
    The Fox and the Tragic Mask
    The Fox and the Grapes
    The Fox and the Goat
    The Raven and the Fox 
    The Fox and the Crow
    The Wolf and the Fox In the Well
    The Fox and the Young Turkeys
    The fox and the crow
    Lion Share
    The Willow-Wren and the Bear
    
    #### 9 ###
    
    The Louse and the Flea
    The devil and his grandmother
    The old man made young again
    The old beggar-woman
    The tinder-box
    The Saucy Boy
    The brave tin soldier
    The storks
    The buckwheat
    The elderbush
    The elfin hill
    The red shoes
    Holger Danske
    By the almshouse window
    The old street lamp
    The old house
    The old grave-stone
    The metal pig
    The Greedy Old Woman
    Shabarsha
    Go I Know Not Where, Bring I Know Not What
    Elena the Wise
    The Fortune-Teller
    The Master-Smith
    Buttercup
    Big Peter and Little Peter 
    Golden Fish
    Tale of the fisherman and the little fish
    The Boy Bathing
    The Man and His Two Sweethearts
    The Astronomer
    The Old Woman and the Physician
    The Old Man and Death
    The Trumpeter Taken Prisoner
    The Sapient Ass
    The Bald-pate Dupe
    The Old Woman and Empty Cask
    The Brother and Sister
    The old man, the boy and the donkey
    
    #### 10 ###
    
    The Wolf and the Seven Young Kids
    Little Red Riding Hood
    Old Sultan
    The wolf and the man
    The Wolf and the Goat
    Little Red Riding Hood
    The Ass and the Wolf
    Acheria, the Fox
    The Wolf and the Lamb
    The Ass and the Wolf
    The Wolf and the Lamb
    The Wolf and the Crane
    The Wolves and the Sheep
    The Shepherd and the Wolf
    The Wolf and the Sheep
    The Wolf and the Housedog
    The Shepherd and the Dog
    The Wolves and the Sheepdogs
    The Kid and the Wolf
    The Wolf and the Shepherd
    The Wolf and the Goat
    The Blind Man and the Whelp
    The Wolf and the Horse
    The Mother and the Wolf
    The Wolf and the Lion
    The Lamb and the Wolf
    The Shepherd's Boy and the Wolf
    The Wolf and the Lamb
    The Sheep, the Stag, and the Wolf
    The Sheep, the Dog, and the Wolf
    The Wolf and Crane
    The Dog and the Wolf
    The Companions of Ulysses
    The Wolf and the Fox
    The Lion, Wolf, and Fox
    
    #### 11 ###
    
    Hansel and Gretel
    The fisherman and his wife
    Mother Hulda
    The seven ravens
    The Bremen town musicians
    The Elves
    The robber bridegroom
    The godfather
    Godfather Death
    The almond tree
    Roland
    Frederick and Catherine
    Jorinda and Joringel
    Gambling Hansel
    The gold-children
    The poor man and the rich man
    Doctor Know-all
    The spirit in the glass bottle
    The seven Swabians
    Donkey cabbages
    The three brothers
    Odds and ends
    Snow-White and Rose-Red
    The glass coffin
    Sharing joy and sorrow
    The owl
    The moon
    Little Ida's flowers
    Thumbelina
    The little mermaid
    The emperor's new suit
    The goloshes of fortune
    The daisy
    The elf of the rose
    The garden of paradise
    The flying trunk
    Ole-Luk-Oie, the Dream-God
    The angel
    The Nightingale
    Top and Ball
    The ugly duckling
    The fir tree
    The snow queen
    The shepherdess and the sweep
    The bell
    Grandmother
    The darning-needle
    The little match-seller
    The neighbouring families
    Little Tuk
    The shadow
    The drop of water
    The happy family
    The story of a mother
    The shirt-collar
    The flax
    There is a difference
    The loveliest rose in the world
    The story of the year
    On judgment day
    A cheerful temper
    Heartache
    Everything in its proper place
    The goblin and the huckster
    Under the willow tree
    Five peas from a pod
    She was good for nothing
    The last pearl
    At the uttermost parts of the sea
    The moneybox
    A leaf from heaven
    Ib and little Christina
    The Jewish girl
    A string of pearls
    The bottle neck
    The old bachelor's nightcap
    Something
    The last dream of the old oak
    The A-B-C book
    The marsh king's daughter
    The stone of the wise man
    The wind tells about Valdemar Daae and his daughters
    The girl who trod on the loaf
    Ole the tower-keeper
    Two brothers
    The butterfly
    Twelve by the mail
    The beetle
    What the old man does is always right
    The snowman
    In the duck yard
    The snail and the rosebush
    The old church bell
    The silver shilling
    The bond of friendship
    A rose from Homer's grave
    A story
    The Dumb Book
    The snowdrop
    The teapot
    The bird of folklore
    The windmill
    Kept secret but not forgotten
    Aunty
    The toad
    Vänö and Glänö
    The little green ones
    The goblin and the woman
    Peiter, Peter, and Peer
    Which was the happiest?
    The puppet-show man
    The days of the week
    The comet
    Sunshine stories
    Chicken Grethe's family
    What happened to the thistle
    What one can invent
    Luck may lie in a pin
    What the whole family said
    Great-Grandfather
    The most incredible thing
    The great sea serpent
    The gardener and the noble family
    What old Johanne told
    Aunty Toothache
    The flea and the professor
    Alyosha Popovich
    Baba Yaga
    The Swan-Geese
    Fenist the Falcon
    A Cunning Trade
    The Wise Maid and the Seven Robbers
    Vasilisa the Beautiful
    The Old Dame and Her Hen
    The Two Step-Sisters
    Gudbrand on the Hill-side
    The Lassie and Her Godmother
    The Three Aunts
    Gertrude's Bird
    Goosey Grizzel
    The Master Thief
    The Husband Who Was to Mind the House
    Thumbikin
    The Sweetheart In The Wood
    Goody Gainst-the-stream
    The Pancake
    Friends In Life And Death
    The Way Of The World
    Reynard And Chanticleer
    Silly Men And Cunning Wives
    The Father Of The Family
    The Town-mouse And The Fell-mouse
    The Sheep And The Pig Who Set Up House
    The Skipper And Old Nick
    The Fairies
    Little Thumb
    Vardiello
    The Seven Doves
    Parsley
    The Months
    The Golden Root
    The Maiden and the Beast
    Little Snow White
    Basa Jauna, the Wild Man
    The Monkeys and Their Mother
    The Monkey and the Dolphin
    The Shipwreck of Simonides
    Esop and the Will
    The Piper and the Puca
    The Enchanted Shoes
    The Man and his Image
    The Unhappily Married Man
    The Maiden
    The Priest and the Corpse
    The Fortune-tellers
    The Cobbler and the Financier
    Democritus and the People of Abdera
    The Monkey and the Cat
    The Elephant and Jupiter's Ape
    The Spider and the Turtle
    The Fiddler in Hell
    The Nyamatsanes
    The Robbers and the Farm Animals
    The story of the selfish husband
    The Story of the Steward
    Cuội and the Moon
    
    #### 12 ###
    
    Clever Hans
    Clever Else
    The thief and his master
    Hans in luck
    Hans married
    The griffin
    Strong Hans
    Clumsy Hans
    The cripple
    
    #### 13 ###
    
    Baba-Yaga and Puny
    Don't Listen, If You Don't Like
    Why the Sea Is Salt
    East o' the Sun and West o' the Moon
    Boots Who Ate a Match With the Troll
    Boots Who Made the Princess Say, That's A Story
    The Giant Who Had No Heart in His Body
    The Mastermaid
    The Cat on the Dovrefell
    Princess on the Glass Hill
    How One Went Out to Woo
    Taming the Shrew
    The Blue Belt
    Not a Pin to Choose Between Them
    The Three Princesses of Whiteland
    Rich Peter the Pedlar
    Boots and the Troll
    The Lad Who Went to the North Wind
    The Best Wish
    Dapplegrim
    Farmer Weathersky
    Lord Peter 
    The Seven Foals
    The Widow's Son
    Bushy Bride
    Boots and His Brothers
    Doll i' the Grass
    The Lad and the Deil
    Soria Moria Castle
    Tom Totherhouse
    The Golden Palace That Hung In The Air
    The Trolls In Hedale Wood
    Boots And His Crew
    Master Tobacco
    The Companion
    The Honest Penny
    Our Parish Clerk
    The Three Years Without Wages
    Silly Matt
    Little Freddy With His Fiddle
    The Squire's Bride
    Osborn's Pipe
    The Storehouse Key in the Disdaff
    The Cock Who Fell into the Brewing Vat
    Why the Sea is Salt
    
    #### 14 ###
    
    Cat and mouse in partnership
    The Mouse, the Bird, and the Sausage
    Soup from a sausage skewer
    Daughter and Stepdaughter
    The Bat and the Weasels 
    The Lion and the Mouse
    The Lion, the Mouse, and the Fox
    The Mouse, the Frog, and the Hawk
    The Mice and the Weasels
    The Cat and Venus
    The Ant and the Snow
    The Man and the Weasel
    Mustela et Mus
    The Mountain in Labor
    The Battle of the Mice and Weasels
    Mustela et Mus
    The League of Rats
    The Lion and the Mouse
    The City Mouse and the Country Mouse
    The Mouse



```python

```
