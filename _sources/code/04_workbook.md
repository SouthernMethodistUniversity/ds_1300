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

# Exploring and Cleaning a Data Set



 By: Dr. Eric Godat and Dr. Rob Kalescky
 
 Natural Language Toolkit: [Documentation](http://www.nltk.org/)
 
 Reference Text: [Natural Language Processing with Python](http://www.nltk.org/book/)
 


## Setup


These are the basic libraries we will use in for data manipulation (pandas) and math functions (numpy). We will add more libraries as we need them.

As a best practice, it is a good idea to load all your libraries in a single cell at the top of a notebook, however for the purposes of this tutorial we will load them as we go.

```python
import pandas as pd
import numpy as np
```

Load a data file into a pandas DataFrame.

This tutorial was designed around using sets of data you have yourselves in a form like a CSV, TSV, or TXT file.  Feel free to use any set of data, but for now we will use a dataset created from scraping this [Multilingual Folktale Database](http://www.mftd.org/).

This file is a CSV filetype, common for text data, but your data may also be stored as TSV's, TXT's, or other file types.  This will slightly change how you read from Pandas, but the concept is largely the same for the different filetypes.  Just keep this in mind when you see references to CSV.

To proceed, you will need to have this file downloaded and in the same folder as this notebook. Alternatively you can put the full path to the file.  Typically, your program will look for the file with the name you specified in the folder that contains your program unless you give the program a path to follow.

```python
data = pd.read_csv('../data/folktales.csv')
data.head()
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

We can do something similar to count the number of characters in the data chunk, including spaces. If you wanted to exclude whitespaces, you could take the list we made above, join it together and count the length of the resulting string.

```python
data = data.fillna("No Information Provided") #If some of our data is missing, this will replace the blank entries. This is only necessary in some cases

```

```python
data['char_count'] = data['Story'].str.len() ## this also includes spaces, to do it without spaces, you could use something like this: "".join()
data[['Story','char_count']].head()
```

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

We can then sort by the average word length.

```python
data[['Story','avg_word']].sort_values(by='avg_word', ascending=True).head()
```

### Count Stopwords


Stopwords are words that are commonly used and do little to aid in the understanding of the content of a text. There is no universal list of stopwords and they vary on the style, time period and media from which your text came from.  Typically, people choose to remove stopwords from their data, as it adds extra clutter while the words themselves provide little to no insight as to the nature of the data.  For now, we are simply going to count them to get an idea of how many there are.

For this tutorial, we will use the standard list of stopwords provided by the Natural Language Toolkit python library.

```python
import nltk
nltk.download('stopwords')
```

```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
```

To count the number of stopwords in a chunk of data, we make a list of all the words in our data that are also in the stopword list. We can then just take the length of that list and store it in a new column

```python
data['stopwords'] = data['Story'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['Story','stopwords','wordcount']].head()
```

### Other Ways to Count


There are other types of counting we might want to do. These might be more or less relevant depending on the test you are working with.

For completeness, we have put them here but we will skip over them in this tutorial


#### Counting Special Characters


This is really only useful for Twitter or other Internet texts but you could imagine wanting to count quotations or exclamations with something similar.

```python
data['special_char'] = data['Story'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['Story','special_char']].head()
```

#### Counting Numbers


This counts the number of numerical digits in a text, which for strict text may not be helpful, but mostly numerical data will make more use of this.

```python
data['numerics'] = data['Story'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['Story','numerics']].sort_values(by='numerics', ascending=False).head()
```

#### Counting Uppercase


Counting uppercase words could give us an indication of how many sentences or proper nouns are in a text but this is likely too broad to be used to classify either on its own.

```python
data['upper'] = data['Story'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['Story','upper']].head()
```

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

### Remove Punctuation


Here we remove all punctuation from the data. This allows us to focus on the words only as well as assist in matching.

```python
data['Story'] = data['Story'].str.replace('[^\w\s]','')
data['Story'].head()
```

### Remove Stopwords


Similar to what we did earlier when we counted stopwords, we now want to remove the stopwords. We will again use the NLTK list of stopwords but this time keep a list of words that do not appear in the list of stopwords.

```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
```

```python
data['Story'] = data['Story'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['Story'].head()
```

### Remove Frequent Words


If we want to catch common words that might have slipped through the stopword removal, we can build out a list of the most common words remaining in our text.

Here we have built a list of the 10 most common words. Some of these words might actually be relevant to our analysis so it is important to be careful with this method.

```python
freq = pd.Series(' '.join(data['Story']).split()).value_counts()[:10]
freq
```

We now follow the same procedure with which we removed stopwords to remove the most frequent words.

```python
freq = list(freq.index)
data['Story'] = data['Story'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Story'].head()
```

### Remove Rare Words


By analogy, we can remove the most rare words. Some of these are strange misspellings or [hapax legomenon](https://en.wikipedia.org/wiki/Hapax_legomenon) (one off words that don't appear anywhere else in the text).

```python
freq = pd.Series(' '.join(data['Story']).split()).value_counts()[-10:]
freq
```

Again, removing words following the same process as the stopword removal.

```python
freq = list(freq.index)
data['Story'] = data['Story'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Story'].head()
```

### Correct Spelling


Misspellings can cause inaccuracies in text analysis. For example, "the" and "teh" are likely intended to be the same word and a reader might gloss over that typo, but a computer would view them as distinct.

To help address this we will leverage the [TextBlob package](https://textblob.readthedocs.io/en/dev/) to check the spelling.

Since this can be slow and often of questionable usefulness, we have limited the check to the first 5 rows of our DataFrame.

It is also worth keeping in mind that, much like autocorrect on your phone, the spellchecking here is not going to be perfectly accurate and could result in just as many errors as it fixes (especially if you are working on text from edited or published sources).

```python
from textblob import TextBlob
data['Story'][:5].apply(lambda x: str(TextBlob(x).correct()))
```

### Tokenization


Tokenization is the process of splitting up a block of text into a sequence of words or sentences.

For those familiar with R and the Tidyverse, this would be referred to as unnesting tokens

Here we show all the tokenized words from the first data chunk in our Dataframe.

```python
#import nltk
#nltk.download('punkt')
```

```python
from textblob import TextBlob

TextBlob(data['Story'][0]).words
```

## Stemming


Stemming is the process of removing suffices, like "ed" or "ing".

We will use another standard NLTK package, PorterStemmer, to do the stemming.

```python
from nltk.stem import PorterStemmer
st = PorterStemmer()
data['Story'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
```

As we can see "wonderful" became "wonder", which could help an analysis, but "deathbed" became "deathb" which is less helpful.


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

## Visualizing Text


Now, we are going to make a word cloud based on our data set. We're going to use the [wordcloud](https://amueller.github.io/word_cloud/) package to help us do that.

```python
import wordcloud
```

```python
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format ='retina'

stop_words = set(stop)
#stop_words.update(["saw"])
```

If you want to update the stopwords after you see the wordcloud, type them into the empty list above and remove the # sign.

```python
word = " ".join(data['Story'])
wordcloud = WordCloud(stopwords=stop, background_color="black").generate(word)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

The above code creates a wordcloud based off all the words (except for stop words) in all of the stories. While this can be fun, it may not be as interesting as a wordcloud for a single story. Let's compare:

```python
for i in range(0, 5):
    print('Title: {}'.format(data['Title'][i]))
    word = data['Story'][i]
    wordcloud = WordCloud(stopwords=stop, background_color="white").generate(word)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
```

# Wrapping Up


At this point we have a several options for cleaning and structuring our text data. We learned how to load data, do basic cleaning and start to analyze our data with some simple counting methods.
