# Comparing Student Work to Generative AI

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
filename = '../data/ds1300_paragraphs.csv'
data = pd.read_csv(filename)
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
      <th>Type</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>The saying data is the new gold could not be m...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>The presence of data, artificial intelligence,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>A</td>
      <td>In our current world, the use of data is both ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>In my everyday life I do not really take notic...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>A</td>
      <td>The advancement of the human race has been lin...</td>
    </tr>
  </tbody>
</table>
</div>



Here we can see all the information available to us from the file in the form of a Pandas DataFrame. For the remainder of this tutorial, we will focus primarily on the full text of each data chunk, which we will name the *Text* column.  With your data set this is likely to be something very different, so feel free to call is something else.

## Counting Words and Characters

The first bit of analysis we might want to do is to count the number of words in one piece of data. To do this we will add a column called *wordcount* and write an operation that applies a function to every row of the column.

Unpacking this piece of code, *len(str(x).split(" ")*, tells us what is happening.

For the content of cell *x*, convert it to a string, *str()*, then split that string into pieces at each space, *split()*.

The result of that is a list of all the words in the text and then we can count the length of that list, *len()*.


```python
data['wordcount'] = data['Text'].apply(lambda x: len(str(x).split(" ")))
data
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
      <th>Type</th>
      <th>Text</th>
      <th>wordcount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>The saying data is the new gold could not be m...</td>
      <td>315</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>The presence of data, artificial intelligence,...</td>
      <td>325</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>A</td>
      <td>In our current world, the use of data is both ...</td>
      <td>354</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>In my everyday life I do not really take notic...</td>
      <td>301</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>A</td>
      <td>The advancement of the human race has been lin...</td>
      <td>412</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>A</td>
      <td>Every day we are working with, sharing, and in...</td>
      <td>338</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>A</td>
      <td>One clear interaction of data that we see in o...</td>
      <td>304</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>285</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>A</td>
      <td>By living, we create data points. This pre-exi...</td>
      <td>496</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>A</td>
      <td>For me, the intersection between data, AI, and...</td>
      <td>316</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>A</td>
      <td>I just completed my first semester at SMU afte...</td>
      <td>417</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>A</td>
      <td>Data, the collection of information; artificia...</td>
      <td>422</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>A</td>
      <td>Data is everywhere. Even if we don't notice it...</td>
      <td>366</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>A</td>
      <td>In today’s day and age, data is one of the mos...</td>
      <td>301</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>A</td>
      <td>I use computers and algorithms everyday of my ...</td>
      <td>249</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>A</td>
      <td>Data, artificial intelligence, and computing a...</td>
      <td>352</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>A</td>
      <td>The use of data is everywhere as soon as you o...</td>
      <td>263</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>A</td>
      <td>Before this class data science meant nothing t...</td>
      <td>312</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>236</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>330</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>A</td>
      <td>As an avid user of social media, data science ...</td>
      <td>244</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>A</td>
      <td>The use of artificial intelligence, computing,...</td>
      <td>313</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>317</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>A</td>
      <td>Data, artificial intelligence, and computing a...</td>
      <td>303</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>432</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>321</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>229</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>300</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>B</td>
      <td>The convergence of data, artificial intelligen...</td>
      <td>304</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>248</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>B</td>
      <td>The integration of data, artificial intelligen...</td>
      <td>288</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>373</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>196</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>B</td>
      <td>In today's interconnected world, the intersect...</td>
      <td>348</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>514</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>185</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>B</td>
      <td>In today's technologically advanced world, the...</td>
      <td>344</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>343</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>B</td>
      <td>In the modern age, the intersection of data, a...</td>
      <td>352</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>178</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>377</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>498</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>247</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>408</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45</td>
      <td>B</td>
      <td>In today's interconnected world, the intersect...</td>
      <td>359</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>B</td>
      <td>Using technology on a daily basis can bring nu...</td>
      <td>126</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>B</td>
      <td>The triad of data, artificial intelligence, an...</td>
      <td>286</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>B</td>
      <td>In the rapidly evolving landscape of technolog...</td>
      <td>330</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>251</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>426</td>
    </tr>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>186</td>
    </tr>
  </tbody>
</table>
</div>



We can do something similar to count the number of characters in the data chunk, including spaces. If you wanted to exclude whitespaces, you could take the list we made above, join it together and count the length of the resulting string.


```python
data['char_count'] = data['Text'].str.len()
data
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
      <th>Type</th>
      <th>Text</th>
      <th>wordcount</th>
      <th>char_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>The saying data is the new gold could not be m...</td>
      <td>315</td>
      <td>1969</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>The presence of data, artificial intelligence,...</td>
      <td>325</td>
      <td>1941</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>A</td>
      <td>In our current world, the use of data is both ...</td>
      <td>354</td>
      <td>2321</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>In my everyday life I do not really take notic...</td>
      <td>301</td>
      <td>1576</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>A</td>
      <td>The advancement of the human race has been lin...</td>
      <td>412</td>
      <td>3065</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>A</td>
      <td>Every day we are working with, sharing, and in...</td>
      <td>338</td>
      <td>1897</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>A</td>
      <td>One clear interaction of data that we see in o...</td>
      <td>304</td>
      <td>1718</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>285</td>
      <td>1710</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>A</td>
      <td>By living, we create data points. This pre-exi...</td>
      <td>496</td>
      <td>2980</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>A</td>
      <td>For me, the intersection between data, AI, and...</td>
      <td>316</td>
      <td>1645</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>A</td>
      <td>I just completed my first semester at SMU afte...</td>
      <td>417</td>
      <td>2438</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>A</td>
      <td>Data, the collection of information; artificia...</td>
      <td>422</td>
      <td>2614</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>A</td>
      <td>Data is everywhere. Even if we don't notice it...</td>
      <td>366</td>
      <td>1959</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>A</td>
      <td>In today’s day and age, data is one of the mos...</td>
      <td>301</td>
      <td>1853</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>A</td>
      <td>I use computers and algorithms everyday of my ...</td>
      <td>249</td>
      <td>1358</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>A</td>
      <td>Data, artificial intelligence, and computing a...</td>
      <td>352</td>
      <td>1853</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>A</td>
      <td>The use of data is everywhere as soon as you o...</td>
      <td>263</td>
      <td>1393</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>A</td>
      <td>Before this class data science meant nothing t...</td>
      <td>312</td>
      <td>1665</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>236</td>
      <td>1392</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>330</td>
      <td>1748</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>A</td>
      <td>As an avid user of social media, data science ...</td>
      <td>244</td>
      <td>1385</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>A</td>
      <td>The use of artificial intelligence, computing,...</td>
      <td>313</td>
      <td>1862</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>317</td>
      <td>1971</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>A</td>
      <td>Data, artificial intelligence, and computing a...</td>
      <td>303</td>
      <td>1703</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>432</td>
      <td>3033</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>321</td>
      <td>2289</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>229</td>
      <td>1516</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>300</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>B</td>
      <td>The convergence of data, artificial intelligen...</td>
      <td>304</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>248</td>
      <td>1680</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>B</td>
      <td>The integration of data, artificial intelligen...</td>
      <td>288</td>
      <td>1843</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>373</td>
      <td>2637</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>196</td>
      <td>1351</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>B</td>
      <td>In today's interconnected world, the intersect...</td>
      <td>348</td>
      <td>2446</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>514</td>
      <td>3155</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>185</td>
      <td>1137</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>B</td>
      <td>In today's technologically advanced world, the...</td>
      <td>344</td>
      <td>2436</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>343</td>
      <td>2353</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>B</td>
      <td>In the modern age, the intersection of data, a...</td>
      <td>352</td>
      <td>2442</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>178</td>
      <td>1184</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>377</td>
      <td>2601</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>498</td>
      <td>2953</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>247</td>
      <td>1835</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>408</td>
      <td>2855</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45</td>
      <td>B</td>
      <td>In today's interconnected world, the intersect...</td>
      <td>359</td>
      <td>2467</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>B</td>
      <td>Using technology on a daily basis can bring nu...</td>
      <td>126</td>
      <td>869</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>B</td>
      <td>The triad of data, artificial intelligence, an...</td>
      <td>286</td>
      <td>1852</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>B</td>
      <td>In the rapidly evolving landscape of technolog...</td>
      <td>330</td>
      <td>2292</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>251</td>
      <td>1570</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>426</td>
      <td>2815</td>
    </tr>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>186</td>
      <td>1268</td>
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
data['avg_word'] = data['Text'].apply(lambda x: avg_word(x))
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
      <th>Type</th>
      <th>Text</th>
      <th>wordcount</th>
      <th>char_count</th>
      <th>avg_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>The saying data is the new gold could not be m...</td>
      <td>315</td>
      <td>1969</td>
      <td>5.227848</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>The presence of data, artificial intelligence,...</td>
      <td>325</td>
      <td>1941</td>
      <td>4.990741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>A</td>
      <td>In our current world, the use of data is both ...</td>
      <td>354</td>
      <td>2321</td>
      <td>5.486034</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>In my everyday life I do not really take notic...</td>
      <td>301</td>
      <td>1576</td>
      <td>4.253333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>A</td>
      <td>The advancement of the human race has been lin...</td>
      <td>412</td>
      <td>3065</td>
      <td>6.360577</td>
    </tr>
  </tbody>
</table>
</div>



We can then sort by the average word length.


```python
data.sort_values(by='avg_word', ascending=True)
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
      <th>Type</th>
      <th>Text</th>
      <th>wordcount</th>
      <th>char_count</th>
      <th>avg_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>A</td>
      <td>For me, the intersection between data, AI, and...</td>
      <td>316</td>
      <td>1645</td>
      <td>4.235669</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>A</td>
      <td>In my everyday life I do not really take notic...</td>
      <td>301</td>
      <td>1576</td>
      <td>4.253333</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>A</td>
      <td>Data, artificial intelligence, and computing a...</td>
      <td>352</td>
      <td>1853</td>
      <td>4.273504</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>330</td>
      <td>1748</td>
      <td>4.300000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>A</td>
      <td>The use of data is everywhere as soon as you o...</td>
      <td>263</td>
      <td>1393</td>
      <td>4.316794</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>A</td>
      <td>Before this class data science meant nothing t...</td>
      <td>312</td>
      <td>1665</td>
      <td>4.353698</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>A</td>
      <td>Data is everywhere. Even if we don't notice it...</td>
      <td>366</td>
      <td>1959</td>
      <td>4.355191</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>A</td>
      <td>I use computers and algorithms everyday of my ...</td>
      <td>249</td>
      <td>1358</td>
      <td>4.475806</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>A</td>
      <td>Every day we are working with, sharing, and in...</td>
      <td>338</td>
      <td>1897</td>
      <td>4.615385</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>A</td>
      <td>Data, artificial intelligence, and computing a...</td>
      <td>303</td>
      <td>1703</td>
      <td>4.623762</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>A</td>
      <td>One clear interaction of data that we see in o...</td>
      <td>304</td>
      <td>1718</td>
      <td>4.663366</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>A</td>
      <td>As an avid user of social media, data science ...</td>
      <td>244</td>
      <td>1385</td>
      <td>4.699588</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>A</td>
      <td>I just completed my first semester at SMU afte...</td>
      <td>417</td>
      <td>2438</td>
      <td>4.800000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>42</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>498</td>
      <td>2953</td>
      <td>4.822134</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>236</td>
      <td>1392</td>
      <td>4.910638</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>A</td>
      <td>The use of artificial intelligence, computing,...</td>
      <td>313</td>
      <td>1862</td>
      <td>4.967949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>A</td>
      <td>The presence of data, artificial intelligence,...</td>
      <td>325</td>
      <td>1941</td>
      <td>4.990741</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>A</td>
      <td>By living, we create data points. This pre-exi...</td>
      <td>496</td>
      <td>2980</td>
      <td>4.995976</td>
    </tr>
    <tr>
      <th>34</th>
      <td>35</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>514</td>
      <td>3155</td>
      <td>5.021033</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>285</td>
      <td>1710</td>
      <td>5.021127</td>
    </tr>
    <tr>
      <th>35</th>
      <td>36</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>185</td>
      <td>1137</td>
      <td>5.151351</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>A</td>
      <td>In today’s day and age, data is one of the mos...</td>
      <td>301</td>
      <td>1853</td>
      <td>5.159468</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>A</td>
      <td>Data, the collection of information; artificia...</td>
      <td>422</td>
      <td>2614</td>
      <td>5.209026</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>A</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>317</td>
      <td>1971</td>
      <td>5.220820</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>The saying data is the new gold could not be m...</td>
      <td>315</td>
      <td>1969</td>
      <td>5.227848</td>
    </tr>
    <tr>
      <th>48</th>
      <td>49</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>251</td>
      <td>1570</td>
      <td>5.258964</td>
    </tr>
    <tr>
      <th>30</th>
      <td>31</td>
      <td>B</td>
      <td>The integration of data, artificial intelligen...</td>
      <td>288</td>
      <td>1843</td>
      <td>5.370242</td>
    </tr>
    <tr>
      <th>46</th>
      <td>47</td>
      <td>B</td>
      <td>The triad of data, artificial intelligence, an...</td>
      <td>286</td>
      <td>1852</td>
      <td>5.401384</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>A</td>
      <td>In our current world, the use of data is both ...</td>
      <td>354</td>
      <td>2321</td>
      <td>5.486034</td>
    </tr>
    <tr>
      <th>49</th>
      <td>50</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>426</td>
      <td>2815</td>
      <td>5.574766</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>229</td>
      <td>1516</td>
      <td>5.624454</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>B</td>
      <td>The convergence of data, artificial intelligen...</td>
      <td>304</td>
      <td>2013</td>
      <td>5.625000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>40</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>178</td>
      <td>1184</td>
      <td>5.657303</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>248</td>
      <td>1680</td>
      <td>5.685259</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>300</td>
      <td>2020</td>
      <td>5.736667</td>
    </tr>
    <tr>
      <th>37</th>
      <td>38</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>343</td>
      <td>2353</td>
      <td>5.777457</td>
    </tr>
    <tr>
      <th>44</th>
      <td>45</td>
      <td>B</td>
      <td>In today's interconnected world, the intersect...</td>
      <td>359</td>
      <td>2467</td>
      <td>5.787879</td>
    </tr>
    <tr>
      <th>40</th>
      <td>41</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>377</td>
      <td>2601</td>
      <td>5.818898</td>
    </tr>
    <tr>
      <th>50</th>
      <td>51</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>186</td>
      <td>1268</td>
      <td>5.822581</td>
    </tr>
    <tr>
      <th>38</th>
      <td>39</td>
      <td>B</td>
      <td>In the modern age, the intersection of data, a...</td>
      <td>352</td>
      <td>2442</td>
      <td>5.829132</td>
    </tr>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>B</td>
      <td>Using technology on a daily basis can bring nu...</td>
      <td>126</td>
      <td>869</td>
      <td>5.873016</td>
    </tr>
    <tr>
      <th>32</th>
      <td>33</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>196</td>
      <td>1351</td>
      <td>5.897959</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>432</td>
      <td>3033</td>
      <td>5.931350</td>
    </tr>
    <tr>
      <th>43</th>
      <td>44</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>408</td>
      <td>2855</td>
      <td>5.941606</td>
    </tr>
    <tr>
      <th>47</th>
      <td>48</td>
      <td>B</td>
      <td>In the rapidly evolving landscape of technolog...</td>
      <td>330</td>
      <td>2292</td>
      <td>5.948485</td>
    </tr>
    <tr>
      <th>33</th>
      <td>34</td>
      <td>B</td>
      <td>In today's interconnected world, the intersect...</td>
      <td>348</td>
      <td>2446</td>
      <td>5.962963</td>
    </tr>
    <tr>
      <th>31</th>
      <td>32</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>373</td>
      <td>2637</td>
      <td>5.986737</td>
    </tr>
    <tr>
      <th>36</th>
      <td>37</td>
      <td>B</td>
      <td>In today's technologically advanced world, the...</td>
      <td>344</td>
      <td>2436</td>
      <td>5.991379</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>321</td>
      <td>2289</td>
      <td>6.033846</td>
    </tr>
    <tr>
      <th>42</th>
      <td>43</td>
      <td>B</td>
      <td>The intersection of data, artificial intellige...</td>
      <td>247</td>
      <td>1835</td>
      <td>6.357430</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>A</td>
      <td>The advancement of the human race has been lin...</td>
      <td>412</td>
      <td>3065</td>
      <td>6.360577</td>
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
data['Text'] = data['Text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Text'].head()
```




    0    the saying data is the new gold could not be m...
    1    the presence of data, artificial intelligence,...
    2    in our current world, the use of data is both ...
    3    in my everyday life i do not really take notic...
    4    the advancement of the human race has been lin...
    Name: Text, dtype: object



### Remove Punctuation

Here we remove all punctuation from the data. This allows us to focus on the words only as well as assist in matching.


```python
data['Text'] = data['Text'].str.replace('[^\w\s]','')
data['Text'].head()
```

    /tmp/ipykernel_1623401/1947507549.py:1: FutureWarning: The default value of regex will change from True to False in a future version.
      data['Text'] = data['Text'].str.replace('[^\w\s]','')





    0    the saying data is the new gold could not be m...
    1    the presence of data artificial intelligence a...
    2    in our current world the use of data is both t...
    3    in my everyday life i do not really take notic...
    4    the advancement of the human race has been lin...
    Name: Text, dtype: object



### Remove Stopwords

Stopwords are words that are commonly used and do little to aid in the understanding of the content of a text. There is no universal list of stopwords and they vary on the style, time period and media from which your text came from.  Typically, people choose to remove stopwords from their data, as it adds extra clutter while the words themselves provide little to no insight as to the nature of the data.  For now, we are simply going to count them to get an idea of how many there are.

For this tutorial, we will use the standard list of stopwords provided by the Natural Language Toolkit python library.


```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['Text'] = data['Text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['Text'].head()
```




    0    saying data new gold could accurate current so...
    1    presence data artificial intelligence computin...
    2    current world use data interesting manipulated...
    3    everyday life really take notice intersection ...
    4    advancement human race linked remarkable progr...
    Name: Text, dtype: object



### Remove Frequent Words

If we want to catch common words that might have slipped through the stopword removal, we can build out a list of the most common words remaining in our text.

Here we have built a list of the 10 most common words. Some of these words might actually be relevant to our analysis so it is important to be careful with this method.


```python
freq = pd.Series(' '.join(data['Text']).split()).value_counts()[:10]
freq
```




    data            363
    algorithms      167
    ai              116
    computing       111
    use             101
    artificial       93
    intelligence     91
    lives            68
    online           67
    way              66
    dtype: int64



We now follow the same procedure with which we removed stopwords to remove the most frequent words.

## Lemmatization

Lemmatization is often a more useful approach than stemming because it leverages an understanding of the word itself to convert the word back to its root word. However, this means lemmatization is less aggressive than stemming (probably a good thing).


```python
import nltk
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to /users/egodat/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!





    True




```python
from textblob import Word
data['Text'] = data['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Text'].head()
```




    0    saying data new gold could accurate current so...
    1    presence data artificial intelligence computin...
    2    current world use data interesting manipulated...
    3    everyday life really take notice intersection ...
    4    advancement human race linked remarkable progr...
    Name: Text, dtype: object



At this point we have a several options for cleaning and structuring our text data. The next section will focus on more advanced ways to study text analytically.

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
tf1 = (data['Text']).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

## Term Frequency Definition
#tf1 = (data['Text'][0:5]).apply(lambda x: (pd.value_counts(x.split(" ")))/len(x.split(" "))).sum(axis = 0).reset_index() 

## Log Scaled Definition
#tf1 = (data['Text'][0:10]).apply(lambda x: 1.0+np.log(pd.value_counts(x.split(" ")))).sum(axis = 0).reset_index() 

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
      <th>3</th>
      <td>data</td>
      <td>363.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>algorithm</td>
      <td>185.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>life</td>
      <td>125.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>ai</td>
      <td>119.0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>computing</td>
      <td>111.0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>use</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>158</th>
      <td>artificial</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>157</th>
      <td>intelligence</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>way</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>256</th>
      <td>online</td>
      <td>67.0</td>
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
    tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Text'].str.contains(word)])))

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
      <td>content</td>
      <td>46.0</td>
      <td>0.712950</td>
    </tr>
    <tr>
      <th>1</th>
      <td>algorithm</td>
      <td>185.0</td>
      <td>0.040005</td>
    </tr>
    <tr>
      <th>2</th>
      <td>company</td>
      <td>25.0</td>
      <td>1.159237</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data</td>
      <td>363.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>interest</td>
      <td>36.0</td>
      <td>0.376478</td>
    </tr>
    <tr>
      <th>5</th>
      <td>new</td>
      <td>24.0</td>
      <td>0.753772</td>
    </tr>
    <tr>
      <th>6</th>
      <td>people</td>
      <td>22.0</td>
      <td>1.446919</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pattern</td>
      <td>20.0</td>
      <td>1.159237</td>
    </tr>
    <tr>
      <th>8</th>
      <td>example</td>
      <td>34.0</td>
      <td>0.753772</td>
    </tr>
    <tr>
      <th>9</th>
      <td>life</td>
      <td>125.0</td>
      <td>0.040005</td>
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
tf1.sort_values(by='tfidf', ascending=False)[:20]
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
      <th>97</th>
      <td>user</td>
      <td>32.0</td>
      <td>1.159237</td>
      <td>37.095581</td>
    </tr>
    <tr>
      <th>479</th>
      <td>assistant</td>
      <td>35.0</td>
      <td>0.987387</td>
      <td>34.558533</td>
    </tr>
    <tr>
      <th>375</th>
      <td>used</td>
      <td>41.0</td>
      <td>0.840783</td>
      <td>34.472110</td>
    </tr>
    <tr>
      <th>626</th>
      <td>potential</td>
      <td>25.0</td>
      <td>1.366876</td>
      <td>34.171907</td>
    </tr>
    <tr>
      <th>438</th>
      <td>personalized</td>
      <td>50.0</td>
      <td>0.673729</td>
      <td>33.686455</td>
    </tr>
    <tr>
      <th>256</th>
      <td>online</td>
      <td>67.0</td>
      <td>0.497838</td>
      <td>33.355175</td>
    </tr>
    <tr>
      <th>0</th>
      <td>content</td>
      <td>46.0</td>
      <td>0.712950</td>
      <td>32.795691</td>
    </tr>
    <tr>
      <th>443</th>
      <td>device</td>
      <td>33.0</td>
      <td>0.987387</td>
      <td>32.583760</td>
    </tr>
    <tr>
      <th>15</th>
      <td>website</td>
      <td>15.0</td>
      <td>2.140066</td>
      <td>32.100992</td>
    </tr>
    <tr>
      <th>6</th>
      <td>people</td>
      <td>22.0</td>
      <td>1.446919</td>
      <td>31.832218</td>
    </tr>
    <tr>
      <th>251</th>
      <td>make</td>
      <td>54.0</td>
      <td>0.564530</td>
      <td>30.484609</td>
    </tr>
    <tr>
      <th>266</th>
      <td>digital</td>
      <td>38.0</td>
      <td>0.796331</td>
      <td>30.260594</td>
    </tr>
    <tr>
      <th>123</th>
      <td>also</td>
      <td>59.0</td>
      <td>0.497838</td>
      <td>29.372467</td>
    </tr>
    <tr>
      <th>504</th>
      <td>various</td>
      <td>28.0</td>
      <td>1.041454</td>
      <td>29.160708</td>
    </tr>
    <tr>
      <th>28</th>
      <td>route</td>
      <td>25.0</td>
      <td>1.159237</td>
      <td>28.980923</td>
    </tr>
    <tr>
      <th>2</th>
      <td>company</td>
      <td>25.0</td>
      <td>1.159237</td>
      <td>28.980923</td>
    </tr>
    <tr>
      <th>254</th>
      <td>based</td>
      <td>34.0</td>
      <td>0.840783</td>
      <td>28.586628</td>
    </tr>
    <tr>
      <th>144</th>
      <td>information</td>
      <td>65.0</td>
      <td>0.435318</td>
      <td>28.295675</td>
    </tr>
    <tr>
      <th>305</th>
      <td>recommendation</td>
      <td>51.0</td>
      <td>0.530628</td>
      <td>27.062041</td>
    </tr>
    <tr>
      <th>267</th>
      <td>decision</td>
      <td>32.0</td>
      <td>0.840783</td>
      <td>26.905062</td>
    </tr>
  </tbody>
</table>
</div>




```python
tf1.sort_values(by='tfidf', ascending=True)[:20]
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
      <th>249</th>
      <td>v</td>
      <td>2.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data</td>
      <td>363.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>le</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>235</th>
      <td>us</td>
      <td>6.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>113</th>
      <td>u</td>
      <td>40.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>715</th>
      <td>ex</td>
      <td>1.0</td>
      <td>0.103184</td>
      <td>0.103184</td>
    </tr>
    <tr>
      <th>429</th>
      <td>end</td>
      <td>3.0</td>
      <td>0.040005</td>
      <td>0.120016</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>art</td>
      <td>1.0</td>
      <td>0.125163</td>
      <td>0.125163</td>
    </tr>
    <tr>
      <th>155</th>
      <td>act</td>
      <td>3.0</td>
      <td>0.060625</td>
      <td>0.181874</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>tell</td>
      <td>1.0</td>
      <td>0.194156</td>
      <td>0.194156</td>
    </tr>
    <tr>
      <th>219</th>
      <td>go</td>
      <td>12.0</td>
      <td>0.019803</td>
      <td>0.237632</td>
    </tr>
    <tr>
      <th>1144</th>
      <td>id</td>
      <td>2.0</td>
      <td>0.147636</td>
      <td>0.295272</td>
    </tr>
    <tr>
      <th>1727</th>
      <td>tech</td>
      <td>1.0</td>
      <td>0.320908</td>
      <td>0.320908</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>ass</td>
      <td>1.0</td>
      <td>0.348307</td>
      <td>0.348307</td>
    </tr>
    <tr>
      <th>580</th>
      <td>im</td>
      <td>6.0</td>
      <td>0.060625</td>
      <td>0.363748</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>eat</td>
      <td>1.0</td>
      <td>0.376478</td>
      <td>0.376478</td>
    </tr>
    <tr>
      <th>422</th>
      <td>put</td>
      <td>4.0</td>
      <td>0.103184</td>
      <td>0.412737</td>
    </tr>
    <tr>
      <th>428</th>
      <td>ad</td>
      <td>23.0</td>
      <td>0.019803</td>
      <td>0.455460</td>
    </tr>
    <tr>
      <th>569</th>
      <td>ever</td>
      <td>5.0</td>
      <td>0.103184</td>
      <td>0.515921</td>
    </tr>
    <tr>
      <th>964</th>
      <td>intersect</td>
      <td>2.0</td>
      <td>0.268264</td>
      <td>0.536528</td>
    </tr>
  </tbody>
</table>
</div>



It is worth noting that the *sklearn* library has the ability to directly calculate a TD-IDF matrix.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
data_vect = tfidf.fit_transform(data['Text'])

data_vect
```




    <51x1000 sparse matrix of type '<class 'numpy.float64'>'
    	with 5502 stored elements in Compressed Sparse Row format>



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
set1 = data[["Index","Type",'Text']]
set2 = set1
```


```python
#To let us do a "cross join": Every row in one gets matched to all rows in the other
set1['key']=1 
set2['key']=1

similarity = pd.merge(set1, set2, on ='key', suffixes=('_1', '_2')).drop("key", 1)
```

    /tmp/ipykernel_1623401/4055549479.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      set1['key']=1
    /tmp/ipykernel_1623401/4055549479.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      set2['key']=1


### Jaccard Similarity

The first is using a metric called the [Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index). This is just taking the intersection of two sets of things (in our case, words or n-grams) and dividing it by the union of those sets. This gives us a metric for understanding how the word usage compares but doesn't account for repeated words since the union and intersections just take unique words. One advantage though is that we can easily extend the single word similarity to compare bi-grams and other n-grams if we want to examine phrase usage.

$$S_{J}(A,B)=\frac{A \cap B}{A \cup B}$$



```python
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
```


```python
def jaccard(row, n=1):
    
    old=row['Text_1']
    new=row['Text_2']
    
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
      <th>Type_1</th>
      <th>Text_1</th>
      <th>Index_2</th>
      <th>Type_2</th>
      <th>Text_2</th>
      <th>Jaccard_Index_for_1_grams</th>
      <th>Jaccard_Index_for_2_grams</th>
      <th>Intersection</th>
      <th>Union</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>144</td>
      <td>144</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>2</td>
      <td>A</td>
      <td>presence data artificial intelligence computin...</td>
      <td>0.102362</td>
      <td>0.000000</td>
      <td>26</td>
      <td>254</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>3</td>
      <td>A</td>
      <td>current world use data interesting manipulated...</td>
      <td>0.094891</td>
      <td>0.005195</td>
      <td>26</td>
      <td>274</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>4</td>
      <td>A</td>
      <td>everyday life really take notice intersection ...</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>17</td>
      <td>238</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>5</td>
      <td>A</td>
      <td>advancement human race linked remarkable progr...</td>
      <td>0.099042</td>
      <td>0.002257</td>
      <td>31</td>
      <td>313</td>
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
    
    old=row['Text_1']
    new=row['Text_2']
    
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
      <th>Type_1</th>
      <th>Text_1</th>
      <th>Index_2</th>
      <th>Type_2</th>
      <th>Text_2</th>
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
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>144</td>
      <td>144</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>2</td>
      <td>A</td>
      <td>presence data artificial intelligence computin...</td>
      <td>0.102362</td>
      <td>0.000000</td>
      <td>26</td>
      <td>254</td>
      <td>0.268382</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>3</td>
      <td>A</td>
      <td>current world use data interesting manipulated...</td>
      <td>0.094891</td>
      <td>0.005195</td>
      <td>26</td>
      <td>274</td>
      <td>0.300258</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>4</td>
      <td>A</td>
      <td>everyday life really take notice intersection ...</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>17</td>
      <td>238</td>
      <td>0.178220</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>A</td>
      <td>saying data new gold could accurate current so...</td>
      <td>5</td>
      <td>A</td>
      <td>advancement human race linked remarkable progr...</td>
      <td>0.099042</td>
      <td>0.002257</td>
      <td>31</td>
      <td>313</td>
      <td>0.324575</td>
    </tr>
  </tbody>
</table>
</div>



### Visualize Similarity


```python
metric = 'Cosine_Similarity'
#metric = 'Jaccard_Index_for_1_grams'
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

plt.figure(figsize=(20,20))        
plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.show()
```


    
![png](output_74_0.png)
    



```python
data[data['Index']==46]
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
      <th>Type</th>
      <th>Text</th>
      <th>wordcount</th>
      <th>char_count</th>
      <th>avg_word</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45</th>
      <td>46</td>
      <td>B</td>
      <td>using technology daily basis bring numerous be...</td>
      <td>126</td>
      <td>869</td>
      <td>5.873016</td>
    </tr>
  </tbody>
</table>
</div>




```python
similarity[['Type_1','Index_1','Type_2','Index_2',metric]].sort_values(by=metric,ascending=False)
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
      <th>Type_1</th>
      <th>Index_1</th>
      <th>Type_2</th>
      <th>Index_2</th>
      <th>Cosine_Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>A</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2444</th>
      <td>B</td>
      <td>48</td>
      <td>B</td>
      <td>48</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>624</th>
      <td>A</td>
      <td>13</td>
      <td>A</td>
      <td>13</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2548</th>
      <td>B</td>
      <td>50</td>
      <td>B</td>
      <td>50</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1612</th>
      <td>B</td>
      <td>32</td>
      <td>B</td>
      <td>32</td>
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
      <th>2315</th>
      <td>B</td>
      <td>46</td>
      <td>A</td>
      <td>21</td>
      <td>0.109789</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>A</td>
      <td>21</td>
      <td>A</td>
      <td>4</td>
      <td>0.103074</td>
    </tr>
    <tr>
      <th>173</th>
      <td>A</td>
      <td>4</td>
      <td>A</td>
      <td>21</td>
      <td>0.103074</td>
    </tr>
    <tr>
      <th>2298</th>
      <td>B</td>
      <td>46</td>
      <td>A</td>
      <td>4</td>
      <td>0.071892</td>
    </tr>
    <tr>
      <th>198</th>
      <td>A</td>
      <td>4</td>
      <td>B</td>
      <td>46</td>
      <td>0.071892</td>
    </tr>
  </tbody>
</table>
<p>2601 rows × 5 columns</p>
</div>




```python
similarity[similarity[metric]>0.05][['Type_1','Index_1','Type_2','Index_2',metric]].sort_values(by=metric,ascending=False)
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
      <th>Type_1</th>
      <th>Index_1</th>
      <th>Type_2</th>
      <th>Index_2</th>
      <th>Cosine_Similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>1</td>
      <td>A</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2444</th>
      <td>B</td>
      <td>48</td>
      <td>B</td>
      <td>48</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>624</th>
      <td>A</td>
      <td>13</td>
      <td>A</td>
      <td>13</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2548</th>
      <td>B</td>
      <td>50</td>
      <td>B</td>
      <td>50</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1612</th>
      <td>B</td>
      <td>32</td>
      <td>B</td>
      <td>32</td>
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
      <th>2315</th>
      <td>B</td>
      <td>46</td>
      <td>A</td>
      <td>21</td>
      <td>0.109789</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>A</td>
      <td>21</td>
      <td>A</td>
      <td>4</td>
      <td>0.103074</td>
    </tr>
    <tr>
      <th>173</th>
      <td>A</td>
      <td>4</td>
      <td>A</td>
      <td>21</td>
      <td>0.103074</td>
    </tr>
    <tr>
      <th>2298</th>
      <td>B</td>
      <td>46</td>
      <td>A</td>
      <td>4</td>
      <td>0.071892</td>
    </tr>
    <tr>
      <th>198</th>
      <td>A</td>
      <td>4</td>
      <td>B</td>
      <td>46</td>
      <td>0.071892</td>
    </tr>
  </tbody>
</table>
<p>2601 rows × 5 columns</p>
</div>



## Using TF-IDF and Machine Learning

This is significantly more advanced than the rest of the tutorial. This takes the TF-IDF matrix and applies a [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering). This groups the texts into clusters of similar terms from the TF-IDF matrix. This algorithm randomly seeds X "means", the values are then clustered into the nearest mean. The centroid of the values in each cluster then becomes the new mean and the process repeats until a convergence is reached.


```python
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

groups = 2

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
tf_idf_matrix = tfidf.fit_transform(data['Text'])

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


    
![png](output_80_0.png)
    



```python
tfidf_test = tf1.sort_values(by='tfidf', ascending=False)[:1000]
```


```python
title_groups = np.transpose([labels,data['Type'],data['Text']])
```

These are the titles of the texts in each cluster. Keep in mind that each time you run the algorithm, the randomness in generating the initial means will result in different clusters.


```python
for i in range(len(title_groups)):
    data.loc[i,'Group'] = title_groups[i][0]
```


```python
grp=0
data[data['Group']==grp]['Text']
```




    0     saying data new gold could accurate current so...
    1     presence data artificial intelligence computin...
    3     everyday life really take notice intersection ...
    5     every day working sharing interacting data art...
    6     one clear interaction data see everyday life u...
    7     intersection data artificial intelligence comp...
    8     living create data point preexisted idea data ...
    9     intersection data ai computing affect constant...
    10    completed first semester smu transferring last...
    12    data everywhere even dont notice phone though ...
    13    today day age data one powerful tool existence...
    14    use computer algorithm everyday life mainly us...
    15    data artificial intelligence computing affect ...
    16    use data everywhere soon open internet educati...
    17    class data science meant nothing didnt really ...
    19    intersection data artificial intelligence comp...
    20    avid user social medium data science play cont...
    23    data artificial intelligence computing affect ...
    45    using technology daily basis bring numerous be...
    Name: Text, dtype: object




```python
for i in range(groups):
    print("")
    print("#### {} ###".format(i))
    print("")
    for el in title_groups:
        if el[0]==i:
            print("{} - {}".format(el[1]," ".join(el[2].split(" ")[:10])))
```

    
    #### 0 ###
    
    A - saying data new gold could accurate current society major multinational
    A - presence data artificial intelligence computing constantly becoming apparent life moment
    A - everyday life really take notice intersection data ai computing something
    A - every day working sharing interacting data artificial intelligence computing common
    A - one clear interaction data see everyday life use social medium
    A - intersection data artificial intelligence computing increasingly large effect daily life
    A - living create data point preexisted idea data opinion human innately
    A - intersection data ai computing affect constant use social medium example
    A - completed first semester smu transferring last fall initial desire largely
    A - data everywhere even dont notice phone though prevalent place see
    A - today day age data one powerful tool existence every aspect
    A - use computer algorithm everyday life mainly use technology daily life
    A - data artificial intelligence computing affect everyday life campus alone computer
    A - use data everywhere soon open internet education social medium navigation
    A - class data science meant nothing didnt really know much clue
    A - intersection data artificial intelligence computing become integral part life especially
    A - avid user social medium data science play content filtered show
    A - data artificial intelligence computing affect modern life daily differently personal
    B - using technology daily basis bring numerous benefit convenience life also
    
    #### 1 ###
    
    A - current world use data interesting manipulated part obtaining information twist
    A - advancement human race linked remarkable progress intersection data artificial intelligence
    A - data collection information artificial intelligence autogenerating programming computing daily task
    A - intersection data artificial intelligence computing affect everyday life even may
    A - use artificial intelligence computing data extreme effect everyday life costumized
    A - intersection data artificial intelligence computing dramatically impacted everyday life moment
    B - intersection data artificial intelligence computing profoundly transformed everyday life permeating
    B - intersection data artificial intelligence ai computing become integral part everyday
    B - intersection data artificial intelligence computing profoundly impacted everyday life shaping
    B - intersection data artificial intelligence computing profound impact daily life digital
    B - convergence data artificial intelligence ai computing significant impact everyday life
    B - intersection data artificial intelligence computing deeply interwoven fabric daily life
    B - integration data artificial intelligence computing transformed way interact world daily
    B - intersection data artificial intelligence computing become integral part everyday life
    B - intersection data artificial intelligence computing profound impact everyday life today
    B - today interconnected world intersection data artificial intelligence ai computing become
    B - intersection data artificial intelligence computing profound impact everyday life constantly
    B - intersection data artificial intelligence ai computing profound impact daily life
    B - today technologically advanced world intersection data artificial intelligence ai computing
    B - intersection data artificial intelligence computing become fundamental part daily life
    B - modern age intersection data artificial intelligence ai computing permeated every
    B - intersection data artificial intelligence computing profound impact everyday life today
    B - intersection data artificial intelligence computing profound impact everyday life reshaping
    B - intersection data artificial intelligence computing profound impact everyday life constantly
    B - intersection data artificial intelligence computing profound farreaching impact daily life
    B - intersection data artificial intelligence computing profoundly impacted everyday life shaping
    B - today interconnected world intersection data artificial intelligence ai computing profound
    B - triad data artificial intelligence computing play instrumental role shaping everyday
    B - rapidly evolving landscape technology intersection data artificial intelligence computing exerts
    B - intersection data artificial intelligence ai computing profound impact daily life
    B - intersection data artificial intelligence computing profound impact everyday life shaping
    B - intersection data artificial intelligence computing profound effect everyday life shaping



```python

```
