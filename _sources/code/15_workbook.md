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

# GeoPandas and Mapping in Python


A large portion of this tutorial was derived from the GeoPandas [user guide](https://geopandas.org/en/stable/docs/user_guide/mapping.html) and [introduction](https://geopandas.org/en/stable/getting_started/introduction.html).


## Getting Started - Boroughs of NYC


One commmon way to visualize and interact with data is to use spatial information in the data to generate maps. In order to do that, we need to build out shapes (called geometries) in our data set.

GeoPandas is an extension of pandas that allows us to include functionality for working with these geometries and perform analyses using that data.


To start we need some data. In this tutorial, we'll use a few built in data sets but we will also load data from external shape files.

```python
import geopandas

%config InlineBackend.figure_format ='retina'
```

### Reading Files

```python
path_to_data = geopandas.datasets.get_path("nybb")
gdf = geopandas.read_file(path_to_data)
gdf
```

### Writing Files

```python
gdf.to_file("my_file.geojson", driver="GeoJSON")
```

### Getting Features

```python
gdf = gdf.set_index("BoroName") #resetting the name of the boroughs as the index
```

Getting the area of the polygon

```python
gdf["area"] = gdf.area
gdf["area"]
```

Getting the boundary of the polygon

```python
gdf['boundary'] = gdf.boundary
gdf['boundary']
```

Getting the centroid (average x and y values) of the polygon

```python
gdf['centroid'] = gdf.centroid
gdf['centroid']
```

### Measuring Distance


Distance from the first point in the data to the centroid

```python
first_point = gdf['centroid'].iloc[0]
gdf['distance'] = gdf['centroid'].distance(first_point)
gdf['distance']
```

Getting the average distance from our values above.

```python
gdf['distance'].mean()
```

## Making Maps


GeoPandas can generate maps based on the geometries in the dataset.

```python
gdf.plot("area", legend=True)
```

You can even make interactive maps that allow you to layer data over an existing map using explore instead of plot

```python
gdf.explore("area", legend=False)
```

We can change the geometry we are plotting to centroids if we only want to plot the center of our shapes

```python
gdf = gdf.set_geometry("centroid")
gdf.plot("area", legend=True)
```

This allows us to plot different layers with both the polygons and the centroids on the same map

```python
ax = gdf["geometry"].plot()
gdf["centroid"].plot(ax=ax, color="black")
```

```python
gdf = gdf.set_geometry("geometry")
```

### Modifying Polygons


We can round out our polygons by creating a "convex hull" around our polygon

```python
gdf["convex_hull"] = gdf.convex_hull
```

```python
ax = gdf["convex_hull"].plot(alpha=.5)  # saving the first plot as an axis and setting alpha (transparency) to 0.5
gdf["boundary"].plot(ax=ax, color="white", linewidth=.5);  # passing the first plot and setting linewitdth to 0.5
```

### Buffering


We might want to add buffers to our polygons.

```python
# buffering the active geometry by 10 000 feet (geometry is already in feet)
gdf["buffered"] = gdf.buffer(10000)

# buffering the centroid geometry by 10 000 feet (geometry is already in feet)
gdf["buffered_centroid"] = gdf["centroid"].buffer(10000)
```

```python
ax = gdf["buffered"].plot(alpha=.5)  # saving the first plot as an axis and setting alpha (transparency) to 0.5
gdf["buffered_centroid"].plot(ax=ax, color="red", alpha=.5)  # passing the first plot as an axis to the second
gdf["boundary"].plot(ax=ax, color="white", linewidth=.5)  # passing the first plot and setting linewitdth to 0.5
```

## Geometry Relations


We can also use our geometries to check for relations between them. For example, we might want to check on intersections or nearby entities.


First we need to isolate the geometry we want to look at - in this case Brooklyn

```python
brooklyn = gdf.loc["Brooklyn", "geometry"]
brooklyn
```

```python
type(brooklyn)
```

Now we can check if any of the other geometries intersect the buffer region around Brooklyn

```python
gdf["buffered"].intersects(brooklyn)
```

This means that only the Bronx fails to intersect with the buffer zone (10,000 ft) around Brooklyn.


We can also look at the centroids and see if the buffered centroids are within their polygon.

```python
gdf["within"] = gdf["buffered_centroid"].within(gdf)
gdf["within"]
```

Plotting to confirm our results (this could be practical when looking at a state like Hawaii to see if the centroid is with in some distance of the shore)

```python
gdf = gdf.set_geometry("buffered_centroid")
ax = gdf.plot("within", legend=True, categorical=True, legend_kwds={'loc': "upper left"})  # using categorical plot and setting the position of the legend
gdf["boundary"].plot(ax=ax, color="black", linewidth=.5);  # passing the first plot and setting linewitdth to 0.5
```

## More on Mapping


We might want to make more types of maps. Here are some examples of ways we can manipulate maps.


### Load Global Data

```python
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
```

```python
cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))
```

```python
world.head()
```

```python
world.plot()
```

### Choropleth Maps


These are maps based on a variable. First we'll filter out some of the data that we don't care about (Antarctica, uninhabitated land).

```python
world = world[(world.pop_est>0) & (world.name!="Antarctica")]
```

Next we divide GDP (groos domestic product) by the population to generate a GDP per capita

```python
world['gdp_per_cap'] = world.gdp_md_est / world.pop_est
```

```python
world.plot(column='gdp_per_cap')
```

### Adding Legends


We can add legends to our map using matplotlib.

```python
import matplotlib.pyplot as plt
```

```python
fig, ax = plt.subplots(1, 1)

world.plot(column='pop_est', ax=ax, legend=True)
```

```python
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots(1, 1)

divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="5%", pad=0.1)

world.plot(column='pop_est', ax=ax, legend=True, cax=cax)

```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

world.plot(column='pop_est',
            ax=ax,
            legend=True,
            legend_kwds={'label': "Population by Country",
                         'orientation': "horizontal"})
```

### Adding Colors


We can use color maps (cmap) to change the color scheme of our map. For more colormap options, check out the [documentation](https://matplotlib.org/2.0.2/users/colormaps.html).

```python
world.plot(column='gdp_per_cap', cmap='gist_rainbow')
```

We can also just plot the boundaries of our polygons

```python
world.boundary.plot()
```

We can use different schemes (here we use quantiles) to classify our map into colors. 

```python
world.plot(column='gdp_per_cap', cmap='OrRd', scheme='quantiles')
```

### Missing Data


GeoPandas will ignore missing data by default but you can force it to fill in that missing data if you want using the missing keywords argument

```python
import numpy as np

world.loc[np.random.choice(world.index, 40), 'pop_est'] = np.nan

world.plot(column='pop_est');
```

```python
world.plot(column='pop_est', missing_kwds={'color': 'lightgrey'});

world.plot(
     column="pop_est",
     legend=True,
     scheme="quantiles",
     figsize=(15, 10),
     missing_kwds={
         "color": "lightgrey",
         "edgecolor": "red",
         "hatch": "///",
         "label": "Missing values",
     },
 );
```

### Other Customizations


We can turn off the axes for a cleaner map

```python
ax = world.plot()

ax.set_axis_off();
```

## Maps with Layers


We can combine maps of different things to make a layered map. Here we've done it with our world map and global cities

```python
cities.plot(marker='*', color='green', markersize=5)

cities = cities.to_crs(world.crs)
```

```python
base = world.plot(color='white', edgecolor='black')

cities.plot(ax=base, marker='o', color='red', markersize=5)
```

## Using Pandas and GeoPandas

```python
import pandas as pd
```

### Read in Shape File


When reading in a shape file, you actually need the reference files in the same directory as the shape file. This means you need a .shp as well as .shx, .dpf, and .prj files with the same names.

```python
usa = geopandas.read_file('../data/s_22mr22.shp')

usa
```

```python
usa['geometry'][42]
```

```python
usa[usa['STATE'] == 'TX'].plot();
```

### Load Pandas


Here we pull in our state facts data set to combine it with the state geometry data.

```python
facts = pd.read_csv('../data/state_facts.tsv', delimiter='\t')
```

```python
facts['USPS_code']=facts['USPS_code'].apply(lambda x: x.strip()) #trim off whitespace to make the abbreviations match
```

### Merge Pandas and GeoPandas

```python
merged = pd.merge(facts,usa, left_on='USPS_code', right_on='STATE', how='left')
```

```python
gdf = geopandas.GeoDataFrame(merged)
```

Now we can filter using conditions like we are familar with when working with pandas.

```python
gdf[gdf['Pop_2020']>5000000].plot()
```
