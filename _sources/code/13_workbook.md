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

# Optimization Introduction

Optimization problems attempt to solve for the optimal solution to a given problem that in general have:
1. An **objective** function that should be taken to its maximum or minimum.
2. A **set of constraints** that set boundaries for the problem.


## Knapsack Problems

A very common class of optimiztion problems are know generally known as **knapsack problems**. At their most basic these problems have datasets composed items with values and weights (costs). Examples could include a grocery shopping list maximizing what you can get for a certain amount of money or diversifying a stock portfolio to maximize the total value, or even what burglar should choose to carry out a house. 


### Greedy Algorithms

Greedy algorithms find approximate solutions, but are generaly fast at providing *a* solution. However, the solution provided may not be the optimal solution.

```python
import numpy as np
import pandas as pd
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
datasets = {'small': (50, "small"), 'large': (100, "large"), 'very_large': (190, "very_large")}
```

```python
dataset = "small"
size = datasets[dataset][0]
data = pd.read_csv("../data/{}.csv".format(datasets[dataset][1]), names=["Value", "Weight"])
data['Name'] = np.arange(len(data))
data = data[['Name', 'Value', 'Weight']]
data.head()
```

```python
names = data.Name.to_list()
values = data.Value.to_list()
weights = data.Weight.to_list()
```

```python
class Item(object):
    def __init__(self, n, v, w):
        self._name = n
        self._value = v
        self._weight = w
    def get_name(self):
        return self._name
    def get_value(self):
        return self._value
    def get_weight(self):
        return self._weight
    def __str__(self):
        return f'<{self._name}, {self._value}, {self._weight}>'

def value(item):
    return item.get_value()

def weight_inverse(item):
    return 1.0/item.get_weight()

def density(item):
    return item.get_value()/item.get_weight()

def pretty_print(result, total_value):
    names = []
    values = []
    weights = []
    for r in result:
        names.append(r.get_name())
        values.append(r.get_value())
        weights.append(r.get_weight())
    d = {'Name': names, 'Value': values, 'Weight': weights}
    df = pd.DataFrame.from_dict(d)
    display(df.head(len(result)))
    print("Total Value: {}".format(locale.currency(total_value, grouping=True)))

def build_items(names, values, weights):
    Items = []
    for i in range(len(values)):
        Items.append(Item(names[i], values[i], weights[i]))
    return Items
```

```python
items = build_items(names, values, weights)
```

```python
def greedy(items, max_weight, key_function):
    """Assumes items a list, max_weight >= 0,
       key_function maps elements of items to numbers"""
    items_copy = sorted(items, key=key_function, reverse = True)
    result = []
    total_value, total_weight = 0.0, 0.0
    for i in range(len(items_copy)):
        if (total_weight + items_copy[i].get_weight()) <= max_weight:
            result.append(items_copy[i])
            total_weight += items_copy[i].get_weight()
            total_value += items_copy[i].get_value()
    pretty_print(result, total_value)
```

```python
%%time
greedy(items, size, value)
```

```python
%%time
greedy(items, size, weight_inverse)
```

```python
%%time
greedy(items, size, density)
```

### Optimal Solution

The optimization case above is specifically called a **0/1 knapsack problem** and is defined by:
1. Each item has a value and a weight.
2. The solution can only be found for a finite weight.
3. A list (I) of finite length contains all the items.
4. A list (V) of finite length contains all the selected items.
5. Maximize:


$$
\sum_{i=0}^{n-1}V_{i}I_{i}^{value}
$$


$$
\sum_{i=0}^{n-1}V_{i}I_{i}^{weight} \leq w
$$


To do this:
1. Get all possible combinations of items.
2. Remove all sets that violate the constraints.
3. Sort and chose the set with the maximal value.

Note that this method quickly becomes intractable for large numbers of items.

```python
def get_binary_rep(n, num_digits):
    result = ''
    while n > 0:
        result = str(n%2) + result
        n = n//2
    if len(result) > num_digits:
        raise ValueError('not enough digits')
    for i in range(num_digits - len(result)):
        result = '0' + result
    return result

def gen_powerset(L):
    powerset = []
    for i in range(0, 2**len(L)):
        bin_str = get_binary_rep(i, len(L))
        subset = []
        for j in range(len(L)):
            if bin_str[j] == '1':
                subset.append(L[j])
        powerset.append(subset)
    return powerset

def choose_best(pset, max_weight, get_val, get_weight):
    best_val = 0.0
    best_set = None
    for items in pset:
        items_val = 0.0
        items_weight = 0.0
        for item in items:
            items_val += get_val(item)
            items_weight += get_weight(item)
        if items_weight <= max_weight and items_val > best_val:
            best_val = items_val
            best_set = items
    return (best_set, best_val)
```

```python
def test_best(items, max_weight = 20):
    pset = gen_powerset(items)
    taken, val = choose_best(pset, max_weight, Item.get_value, Item.get_weight)
    pretty_print(taken, val)
```

```python
%%time
test_best(items, size)
```

## Dynamic Programming

Broadly, dynamic programming is a set of programming methods for solving problems where the problem has the characteristics of **optimal substructure** and **overlapping subproblems**. The former means that the problem can be decomposed into smaller problems and that the finding the optimal solutions to the smaller problems will lead to a global optimal solution for the problem. The latter means that the smaller problems are all formulated in the same way, i.e. they are are all the same.


### Fibonacci Sequences

```python
def fib(n):
    """Assumes n is an int >= 0
       Returns Fibonacci of n"""
    if n == 0 or n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
```

```python
%%time
fib(40)
```

The above version of the algorithm contantly reevaluates `fib` rather than reuse values it has already calculated. This is simple implementation, but doing just `fib(120)` would end up calling `fib` 8,670,007,398,507,948,658,051,921 times. If each call only took one nanosecond we'd get our solution in about 250,000 years.

Below we employ **memoziation**, which is basically just remembering work that has come before.

```python
def fib_memo(n, memo = None):
    """Assumes n is an int >= 0, memo used only by recursive calls
       Returns Fibonacci of n"""
    if memo == None:
        memo = {}
    if n == 0 or n == 1:
        return 1
    try:
        return memo[n]
    except KeyError:
        result = fib_memo(n-1, memo) + fib_memo(n-2, memo)
        memo[n] = result
        return result
```

```python
%%time
fib_memo(40)
```

### Solving the 0/1 Knapsack Problem Using Dynamic Programming

Thinking back on our previous attempts to solve our knapsack problem, the greedy algorithm finds a solution in $log\ n$ time, but that solution may not be optimal, and the brute-force, exhaustive alogirhtm would give use an optimal solution, but in $O(n2^{n})$, i.e. exponetial time. Is there a better way?

Using dynamic programming methods we can optimally search through the space of possible solutions as if it were a tree. In the case of the 0/1 knapsack problem, the tree is what is called a **rooted binary tree**, which is an acyclic directed graph with the following properties:
1. The graph as one root node that has no parent nodes.
2. Other thant the root node, all nodes have exactly on parent.
3. Each node only has two children at most.

For the problem specifically we have:
1. A list that contains all the items.
2. A list that contains all the not yet selected items.
3. The total value of the selected items.
4. The remaining value within which items can be selected.

As each edge connecting the nodes represents whether or not an item is taken these trees are also called **decision trees**.

```python
def max_val(to_consider, avail):
    """Assumes to_consider a list of items, avail a weight
       Returns a tuple of the total value of a solution to the
         0/1 knapsack problem and the items of that solution"""
    if to_consider == [] or avail == 0:
        result = (0, ())
    elif to_consider[0].get_weight() > avail:
        #Explore right branch only
        result = max_val(to_consider[1:], avail)
    else:
        next_item = to_consider[0]
        #Explore left branch
        with_val, with_to_take = max_val(to_consider[1:],
                                     avail - next_item.get_weight())
        with_val += next_item.get_value()
        #Explore right branch
        without_val, without_to_take = max_val(to_consider[1:], avail)
        #Choose better branch
        if with_val > without_val:
            result = (with_val, with_to_take + (next_item,))
        else:
            result = (without_val, without_to_take)
    return result

def dynamic(items, max_weight):
    result = max_val(items, max_weight)
    pretty_print(list(result[1]), result[0])
```

```python
%%time
dynamic(items, size)
```

The above implementation used recursion directly just as with our `fib` implementation. Likewise, memoziation can also be used to improve performance.

```python
def fast_max_val(to_consider, avail, memo = {}):
    """Assumes to_consider a list of items, avail a weight
         memo supplied by recursive calls
       Returns a tuple of the total value of a solution to the
         0/1 knapsack problem and the items of that solution"""
    if (len(to_consider), avail) in memo:
        result = memo[(len(to_consider), avail)]
    elif to_consider == [] or avail == 0:
        result = (0, ())
    elif to_consider[0].get_weight() > avail:
        #Explore right branch only
        result = fast_max_val(to_consider[1:], avail, memo)
    else:
        next_item = to_consider[0]
        #Explore left branch
        with_val, with_to_take =\
                 fast_max_val(to_consider[1:],
                            avail - next_item.get_weight(), memo)
        with_val += next_item.get_value()
        #Explore right branch
        without_val, without_to_take = fast_max_val(to_consider[1:],
                                                avail, memo)
        #Choose better branch
        if with_val > without_val:
            result = (with_val, with_to_take + (next_item,))
        else:
            result = (without_val, without_to_take)
    memo[(len(to_consider), avail)] = result
    return result

def dynamic_memo(items, max_weight):
    result = fast_max_val(items, max_weight)
    pretty_print(list(result[1]), result[0])
```

```python
%%time
dynamic_memo(items, size)
```

## References

1. Example code derived from [Introduction to Computation and Programming Using Python](https://github.com/guttag/Intro-to-Computation-and-Programming).
2. Example data derived from [D-Wave Examples](https://github.com/dwave-examples/knapsack).
