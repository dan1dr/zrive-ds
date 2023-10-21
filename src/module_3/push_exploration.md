# Notebook structure

1) First brute approach, developed before the session on Oct-15th.
2) Best practices applied from the session + improved version. Applied proper set splitting, baseline, features + parameters.

# 1. First approach

## Import libraries and read data


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
import os
```


```python
path = '/home/dan1dr/data/feature_frame.csv'
data = pd.read_csv(path)
```

## Understanding the problem

Develop a ML model that. given a user and product, predicts if the user would purchase it at that moment. Here will explore and select the model we will apply to the PoC. This model will be used to target users and send them a push notification. Relevant info:

- Current push notificiations have an open rate of 5%.
- Focus only on purchases of at least 5 items (shipping cost).
- Use only linear models to speed up the development.
- The result should allow Sales team to select an item from a list and segment the users for triggering that notification.
- Target: expected increase on monthly sales by 2% and uplift of 25% on selected items.

## Filtering and data preparation


```python
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
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>




```python
num_items_ordered = data.groupby('order_id')['outcome'].sum()
filter = num_items_ordered[num_items_ordered >= 5].index

filtered_data = data[data['order_id'].isin(filter)]

print(f"Length initial data: {len(data)}")
print(f"Length filtered data: {len(filtered_data)}\n")

print(f"Unique orders initially: {data['order_id'].nunique()}")
print(f"Unique orders >= 5 items: {filtered_data['order_id'].nunique()}")
```

    Length initial data: 2880549
    Length filtered data: 2163953
    
    Unique orders initially: 3446
    Unique orders >= 5 items: 2603



```python
print(num_items_ordered[num_items_ordered > 5].mean())
print(num_items_ordered[num_items_ordered > 5].median())
```

    12.527332511302918
    11.0


## Feature Engineering

According with previous assignments, we will select only the features that we think are more relevant for our prediction. Will make a few adjustments here and will be iterating along the notebook.

We will create a logistic regression model for the model. From there, we know this model may sensitive to feature scale (keep in mind potential feature scaling if model is poor). Additionaly, for feature selection, we will need to select relevant features to simplify it. The multicollinearity might play an important role, so we will be discarding highly correlated features. To sum up, we might need to create new feature that groups others.

The non-numeric cols will need to be enconded, so we could try applying one-hot encoding.

1. First, we will try to do some manual feature engineering.
2. Later, we will apply Lasso to force some coefficients to be zero and compare to our manual approach
3. Additionally, we might apply Ridge to see also the coefficients obtained (less extreme selection).

## 1) Manual


```python
pd.set_option('display.max_columns', None)
filtered_data.head()
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
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>vendor</th>
      <th>global_popularity</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>2020-10-06 10:50:23</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We will remeber the classification we did in previous notebook:

predicted = ['outcome']
information = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
numerical = ['user_order_seq', 'normalised_price', 'discount_pct', 'global_popularity',
            'count_adults', 'count_children', 'count_babies', 'count_pets', 
            'people_ex_baby', 'days_since_purchase_variant_id', 
            'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
            'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
                'std_days_to_buy_product_type']

categorical = ['product_type', 'vendor']
binary = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
```

From numerical: 
- We will remove count_adults, count_children, count_pets and keep only count_adults, which seems to be highly representative. We will maintain count_babies as correlation it is not that high (0.15) and may provide info.
- We will remove std_days_to_buy_product_type and keep avg_days_to_buy_product_type (highly correlated between themselves. We may do it reversely also)
- We will remove std_days_to_buy_variant_id and keep avg_days_to_buy_variant_id (same thing)

From categorical:
- We will remove vendor and keep product_type (the former has too different values). Later we will apply one-hot encoding

From binary: 
- We would try to use some resampling technique for grouping the 4 into just 1. We will create a column 'any_event' to input value '1' if any of the four cols has a value of 1. In this sense, we address the unbalanced distribution (a bit), we simplify it and also keep the info if any of these events occured



```python
# Remove numericals
numerical_remove = ['count_adults', 'count_children', 'count_babies',
                    'std_days_to_buy_product_type', 'std_days_to_buy_variant_id']
numerical = [col for col in numerical if col not in numerical_remove]

# Remove categoricals
categorical.remove('vendor')

# Create the binary one
filtered_data['any_event'] = filtered_data[['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']].any(axis=1).astype(int)
binary = ['any_event']
# I'm thinking if maybe would be interesting to perform a sume here. If any_event = 2 would be stronger than 1 and so, while also keeping info about events.

```

    /tmp/ipykernel_481/1254836267.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_data['any_event'] = filtered_data[['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']].any(axis=1).astype(int)


Additionally, let's check if order_date is always equal to created_at (maybe some orders are created but not ordered until X days). If so, let's remove order_date (created_at is has also hour and minut info). Maybe hourly data would be crucial for planning the timing of sending notifications.


```python
if len(filtered_data[filtered_data['order_date'] == filtered_data['created_at']]) == len(filtered_data):
    print("ofc")
```


```python
information.remove('order_date')
information
```




    ['variant_id', 'order_id', 'user_id', 'created_at']




```python
cols = information + numerical + categorical + binary + predicted
final_data = filtered_data[cols]
```

Time format encoding for created_at. Later we could inspect if this granularity it is too much and it is better to leave it at day_of_week info


```python
#Verify created at has pandas date_format
final_data['created_at'] = pd.to_datetime(final_data['created_at'])

# Extract year, month, day, and hour as separate features
final_data['year'] = final_data['created_at'].dt.year
final_data['month'] = final_data['created_at'].dt.month
# Extract the day of the week (numerical, 1-7, starting with Monday as 1)
final_data['day_of_week'] = final_data['created_at'].dt.dayofweek + 1
final_data['hour'] = final_data['created_at'].dt.hour
final_data = final_data.drop(columns=['created_at'])
```

    /tmp/ipykernel_481/1071411211.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['created_at'] = pd.to_datetime(final_data['created_at'])
    /tmp/ipykernel_481/1071411211.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['year'] = final_data['created_at'].dt.year
    /tmp/ipykernel_481/1071411211.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['month'] = final_data['created_at'].dt.month
    /tmp/ipykernel_481/1071411211.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['day_of_week'] = final_data['created_at'].dt.dayofweek + 1
    /tmp/ipykernel_481/1071411211.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['hour'] = final_data['created_at'].dt.hour


One-hot encoding for product_type


```python
# Apply one-hot encoding for 'vendor'
# Drop_first = True to get rid of an additional col. Binaries for keeping consistency as any_event
final_data = pd.get_dummies(final_data, columns=['product_type'], prefix='product', drop_first=True).astype(int)

print(final_data.shape)
final_data.head()
```

    (2163953, 80)





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
      <th>variant_id</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>user_order_seq</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>global_popularity</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>any_event</th>
      <th>outcome</th>
      <th>year</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>hour</th>
      <th>product_allpurposecleaner</th>
      <th>product_babyfood12months</th>
      <th>product_babyfood6months</th>
      <th>product_babymilkformula</th>
      <th>product_babytoiletries</th>
      <th>product_bathroomlimescalecleaner</th>
      <th>product_bathshowergel</th>
      <th>product_beer</th>
      <th>product_binbags</th>
      <th>product_bodyskincare</th>
      <th>product_catfood</th>
      <th>product_cereal</th>
      <th>product_cleaningaccessories</th>
      <th>product_coffee</th>
      <th>product_condimentsdressings</th>
      <th>product_cookingingredientsoils</th>
      <th>product_cookingsaucesmarinades</th>
      <th>product_delicatesstainremover</th>
      <th>product_dental</th>
      <th>product_deodorant</th>
      <th>product_dishwasherdetergent</th>
      <th>product_dogfood</th>
      <th>product_driedfruitsnutsseeds</th>
      <th>product_dryingironing</th>
      <th>product_fabricconditionerfreshener</th>
      <th>product_facialskincare</th>
      <th>product_feedingweaning</th>
      <th>product_femininecare</th>
      <th>product_floorcleanerpolish</th>
      <th>product_foodstorage</th>
      <th>product_haircare</th>
      <th>product_handsoapsanitisers</th>
      <th>product_healthcarevitamins</th>
      <th>product_homebaking</th>
      <th>product_householdsundries</th>
      <th>product_jamhoneyspreads</th>
      <th>product_juicesquash</th>
      <th>product_kidsdental</th>
      <th>product_kidssnacks</th>
      <th>product_kitchenovencleaner</th>
      <th>product_kitchenrolltissues</th>
      <th>product_longlifemilksubstitutes</th>
      <th>product_maternity</th>
      <th>product_nappies</th>
      <th>product_nappypants</th>
      <th>product_petcare</th>
      <th>product_pickledfoodolives</th>
      <th>product_premixedcocktails</th>
      <th>product_ricepastapulses</th>
      <th>product_shavinggrooming</th>
      <th>product_snacksconfectionery</th>
      <th>product_softdrinksmixers</th>
      <th>product_superfoodssupplements</th>
      <th>product_tea</th>
      <th>product_tinspackagedfoods</th>
      <th>product_toiletroll</th>
      <th>product_washingcapsules</th>
      <th>product_washingliquidgel</th>
      <th>product_washingpowder</th>
      <th>product_windowglasscleaner</th>
      <th>product_wipescottonwool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>1</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>1</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Train model

Dataset seems to be big enough (2M rows), so we will split the data by i) 80% training set/ 10 validation/ 10 test; later ii) 70/15/15. Note that train_test_split applies a random selection (seed predefined for obtaining same values later on) with no replacement.


```python
# Import model and split method
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Features and target variable
X = final_data.drop(columns='outcome')
y = final_data['outcome']

# Split the data and then split validation/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# We will do the same every time but will test it only against validation set
```


```python
# Initialize model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)
```

    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=2):
    ABNORMAL_TERMINATION_IN_LNSRCH.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
from sklearn.metrics import confusion_matrix, classification_report

# Validate the model
y_val_pred = model.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, y_val_pred))


print("Confusion Matrix:")
print("Watch out: Rows are actual values (N and P), cols are predicted (N and P)")
print(confusion_matrix(y_val, y_val_pred))
```

    Classification Report:


    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99    213182
               1       0.00      0.00      0.00      3213
    
        accuracy                           0.99    216395
       macro avg       0.49      0.50      0.50    216395
    weighted avg       0.97      0.99      0.98    216395
    
    Confusion Matrix:
    Watch out: Rows are actual values (N and P), cols are predicted (N and P)
    [[213182      0]
     [  3213      0]]


    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


Model does not converge with the default number of iterations = 100. Model didn't classified anything as positive! If we try to plot the curves, we will appreciate the classifier it is not a better option than flipping a coin and estimating.


```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

```




    <matplotlib.legend.Legend at 0x7fca6b239d10>




    
![png](push_exploration_files/push_exploration_32_1.png)
    



```python
# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_33_0.png)
    


We will use the feature scaling provided by scikit in order to standardized the features and enable the estimator to learn from them easier. This will substract the mean from the feature and divide by the std.


```python
from sklearn.preprocessing import StandardScaler

X = final_data.drop(columns='outcome')
y = final_data['outcome']

# Feature scaler and apply to X data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Initialize model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>




```python
# Validate the model
y_val_pred = model.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))
```

    Classification Report:


    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))
    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))


                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99    213182
               1       0.00      0.00      0.00      3213
    
        accuracy                           0.99    216395
       macro avg       0.49      0.50      0.50    216395
    weighted avg       0.97      0.99      0.98    216395
    
    Confusion Matrix:
    [[213182      0]
     [  3213      0]]


    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/metrics/_classification.py:1469: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



```python
# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_37_0.png)
    


We see the model it is very good for predicting the negative (no-purchase), but poor for positive (purchases). This may come from the strongly unbalanced classes we had in the dataset.


```python
final_data['outcome'].value_counts()
```




    outcome
    0    2132624
    1      31329
    Name: count, dtype: int64




```python
final_data['any_event'].value_counts()
```




    any_event
    0    2104207
    1      59746
    Name: count, dtype: int64



First, let's usee the weight_class that scikit offers. Then we could think of oversampling the minority class, or undersampling the majority. We have 2M rows, so undersampling might be a good idea since we might have enough data for doing that. Oversampling may disrupt the data and add new ones, but may work also


```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Initialize model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced')

# Train
model.fit(X_train, y_train)
```




<style>#sk-container-id-3 {color: black;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(class_weight=&#x27;balanced&#x27;)</pre></div></div></div></div></div>




```python
# Validate the model
y_val_pred = model.predict(X_val)
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87    213182
               1       0.04      0.61      0.08      3213
    
        accuracy                           0.78    216395
       macro avg       0.52      0.70      0.48    216395
    weighted avg       0.98      0.78      0.86    216395
    
    Confusion Matrix:
    [[166533  46649]
     [  1244   1969]]



```python
# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_44_0.png)
    


Better results with this approach, where we lose precision but gain sensitivity for purchases.

Let's also try luck using UnderSampler from under_sampling library AFTER the split data, for avoiding information leakage (another method for undersampling data)


```python
from imblearn.under_sampling import RandomUnderSampler

X = final_data.drop(columns='outcome')
y = final_data['outcome']

# Feature scaler and apply to X data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Define undersample strategy
undersample = RandomUnderSampler(sampling_strategy='majority')

# Fit and apply the transform to the training data
X_train, y_train = undersample.fit_resample(X_train, y_train)

# Train your model on the undersampled training data
model.fit(X_train, y_train)

# Predict the outcomes for the validation data
y_val_pred = model.predict(X_val)

# Print the classification report for validation results
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.79      0.88    213182
               1       0.04      0.61      0.08      3213
    
        accuracy                           0.78    216395
       macro avg       0.52      0.70      0.48    216395
    weighted avg       0.98      0.78      0.86    216395
    
    Confusion Matrix:
    [[167349  45833]
     [  1251   1962]]



```python
# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_48_0.png)
    


We obtain basically very similar results, so for the shake of simplicity, we will stay with the weight class that scikit provides. Curiously, both results obtain similar curves vs with no balance class for logistic regression. It may be due to the fact that, given the class 1 is so umbalanced, we do not appreciate high differences. However, we can see that in the precision in purchases (60% vs 4%, normal regression accuracy for positive is better) and recall in purchases (0% vs 62%, simple logistic regression does not find any purchase, while the later models do).

Now, with these results, let's pause for a moment and analyze the numbers obtained:

- Precision (True positive / true positive + false positive). Measures the accuracy of positive predictions. If the model predicts a purchase won't happen, it is correct almost always. If it predicts it will happen, only is correct 4% of the time (generating lots of false positives). Remember that current push open rate is around 5%, so it is feasible to be slightly below that. Would be useful to have the purchase rate per opened notification.

- Recall or true positive rate (True positive / true positive + false negative). Measures the ability to identify all positives. If model predicts no purchase, it identifies 77% of the time. If the model predicts a purchase, it identifies 63% of the purchases

- ROC Curve. Knowing that classes are strongly unbalanced, we can appreciate a few things: We start to generate false alarms very quickly, as provided by precision before. When 50% of the alarms are false (FPR at 0.5), we are identifying 80% of the positive alarms. Then the number of false alarms grow very quickly (so basically we will be spamming the user)

- Precision-recall curve. Precision decreases very fast for almost no recall (loss of accuracy of positive predictions when trying to gain sensivity).

We want to have a recall enough to appreciate purchases, and with a reasonable amount of false positives but fixed for avoiding churn rates (which Sales team mentioned it had a huge cost). Since we would like to target those users that are likely to buy those products, we can expect to increase the probability threshold and capture secure notifications. We will be missing positives, but sending those that we are more sure they will be buying.

We see a light peak on the precision-recall curve where recall is around = 0.32, we might try to get the probability threshold from there and try to see how our model performs with that parameter:




```python
precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)

desired_recall = 0.32
closest_recall_idx = np.argmin(np.abs(recall - desired_recall))

# Get the threshold corresponding to that index
threshold_at_desired_recall = thresholds[closest_recall_idx]
print(f"Threshold for Recall = {desired_recall}: {threshold_at_desired_recall}")
```

    Threshold for Recall = 0.32: 0.8416240756662322



```python
custom_threshold = 0.84

# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]
y_val_custom_threshold = (y_val_probs > custom_threshold).astype(int)


print("Classification Report:")
print(classification_report(y_val, y_val_custom_threshold))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_custom_threshold))
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.98      0.98    213182
               1       0.18      0.32      0.23      3213
    
        accuracy                           0.97    216395
       macro avg       0.59      0.65      0.61    216395
    weighted avg       0.98      0.97      0.97    216395
    
    Confusion Matrix:
    [[208529   4653]
     [  2180   1033]]


We see we misscalculated 2k positive values as negative (false negative), and we generated 4k false positives. However, we were right for 1k purchases realised. The precision for purchase was 18%, which compared to our current open rate, seems OK.


```python
custom_threshold = 0.7

# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]
y_val_custom_threshold = (y_val_probs > custom_threshold).astype(int)


print("Classification Report:")
print(classification_report(y_val, y_val_custom_threshold))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_custom_threshold))
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.97      0.98    213182
               1       0.14      0.35      0.20      3213
    
        accuracy                           0.96    216395
       macro avg       0.57      0.66      0.59    216395
    weighted avg       0.98      0.96      0.97    216395
    
    Confusion Matrix:
    [[206260   6922]
     [  2084   1129]]


Now there is the trade-off betweem how many we want to capture and generate as false positive vs the total we are able to dismiss, so the max number for false negatives. As per the problem's statement, we would like to avoid false positives as they might churn – hence lower precision for positives but also lower recall. In this sense, we might select P = 0.84 instead of 0.7

If we want to know what were the weights assigned in my model:


```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Initialize model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced')

# Train
model.fit(X_train, y_train)

# Access the coefficients (weights) of the features
X = final_data.drop(columns='outcome')
coefficients = model.coef_[0]

# Create a DataFrame to associate coefficients with feature names
coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

# Sort the DataFrame by coefficient values (for better visualization)
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)

# Display the coefficients
coefficients_df.head(10)
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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>year</td>
      <td>-0.672157</td>
    </tr>
    <tr>
      <th>12</th>
      <td>avg_days_to_buy_product_type</td>
      <td>-0.607223</td>
    </tr>
    <tr>
      <th>15</th>
      <td>month</td>
      <td>-0.525724</td>
    </tr>
    <tr>
      <th>13</th>
      <td>any_event</td>
      <td>0.516628</td>
    </tr>
    <tr>
      <th>61</th>
      <td>product_nappies</td>
      <td>-0.393418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>normalised_price</td>
      <td>-0.297199</td>
    </tr>
    <tr>
      <th>23</th>
      <td>product_bathroomlimescalecleaner</td>
      <td>0.285365</td>
    </tr>
    <tr>
      <th>38</th>
      <td>product_dishwasherdetergent</td>
      <td>0.263845</td>
    </tr>
    <tr>
      <th>21</th>
      <td>product_babymilkformula</td>
      <td>-0.262905</td>
    </tr>
    <tr>
      <th>72</th>
      <td>product_tinspackagedfoods</td>
      <td>0.249028</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefficients_df.tail(10)
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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>user_id</td>
      <td>-0.017418</td>
    </tr>
    <tr>
      <th>7</th>
      <td>count_pets</td>
      <td>0.016740</td>
    </tr>
    <tr>
      <th>44</th>
      <td>product_feedingweaning</td>
      <td>-0.014295</td>
    </tr>
    <tr>
      <th>19</th>
      <td>product_babyfood12months</td>
      <td>0.009898</td>
    </tr>
    <tr>
      <th>8</th>
      <td>people_ex_baby</td>
      <td>-0.009382</td>
    </tr>
    <tr>
      <th>1</th>
      <td>order_id</td>
      <td>0.008983</td>
    </tr>
    <tr>
      <th>16</th>
      <td>day_of_week</td>
      <td>0.006718</td>
    </tr>
    <tr>
      <th>70</th>
      <td>product_superfoodssupplements</td>
      <td>-0.002507</td>
    </tr>
    <tr>
      <th>67</th>
      <td>product_shavinggrooming</td>
      <td>-0.000317</td>
    </tr>
    <tr>
      <th>6</th>
      <td>global_popularity</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



There a lots of values almost not used for the prediction, so let's try to use Lasso to penalize them and make another predicitons


```python
# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_60_0.png)
    


## 2) Lasso


```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Add L1 penalty and choose an allowed solver
model = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced')

# Train
model.fit(X_train, y_train)

# Predict the outcomes for the validation data
y_val_pred = model.predict(X_val)
```


```python
X = final_data.drop(columns='outcome')

# Print the classification report for validation results
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

coefficients = model.coef_[0]  # Coefficients for the first (and only) class
coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)
coefficients_df.head(10)
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87    213182
               1       0.04      0.61      0.08      3213
    
        accuracy                           0.78    216395
       macro avg       0.52      0.70      0.48    216395
    weighted avg       0.98      0.78      0.86    216395
    
    Confusion Matrix:
    [[166535  46647]
     [  1244   1969]]





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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>avg_days_to_buy_product_type</td>
      <td>-0.772772</td>
    </tr>
    <tr>
      <th>14</th>
      <td>year</td>
      <td>-0.671969</td>
    </tr>
    <tr>
      <th>15</th>
      <td>month</td>
      <td>-0.525529</td>
    </tr>
    <tr>
      <th>13</th>
      <td>any_event</td>
      <td>0.516638</td>
    </tr>
    <tr>
      <th>61</th>
      <td>product_nappies</td>
      <td>-0.477559</td>
    </tr>
    <tr>
      <th>21</th>
      <td>product_babymilkformula</td>
      <td>-0.320373</td>
    </tr>
    <tr>
      <th>60</th>
      <td>product_maternity</td>
      <td>-0.291124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>normalised_price</td>
      <td>-0.270189</td>
    </tr>
    <tr>
      <th>23</th>
      <td>product_bathroomlimescalecleaner</td>
      <td>0.245388</td>
    </tr>
    <tr>
      <th>56</th>
      <td>product_kidssnacks</td>
      <td>-0.242586</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefficients_df.tail(20)
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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>discount_pct</td>
      <td>0.042723</td>
    </tr>
    <tr>
      <th>69</th>
      <td>product_softdrinksmixers</td>
      <td>0.035872</td>
    </tr>
    <tr>
      <th>9</th>
      <td>days_since_purchase_variant_id</td>
      <td>-0.034336</td>
    </tr>
    <tr>
      <th>54</th>
      <td>product_juicesquash</td>
      <td>-0.027826</td>
    </tr>
    <tr>
      <th>27</th>
      <td>product_bodyskincare</td>
      <td>0.027113</td>
    </tr>
    <tr>
      <th>31</th>
      <td>product_coffee</td>
      <td>0.026051</td>
    </tr>
    <tr>
      <th>63</th>
      <td>product_petcare</td>
      <td>0.024966</td>
    </tr>
    <tr>
      <th>44</th>
      <td>product_feedingweaning</td>
      <td>-0.023822</td>
    </tr>
    <tr>
      <th>17</th>
      <td>hour</td>
      <td>0.020988</td>
    </tr>
    <tr>
      <th>67</th>
      <td>product_shavinggrooming</td>
      <td>-0.020007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>user_id</td>
      <td>-0.017428</td>
    </tr>
    <tr>
      <th>7</th>
      <td>count_pets</td>
      <td>0.016734</td>
    </tr>
    <tr>
      <th>52</th>
      <td>product_householdsundries</td>
      <td>0.015235</td>
    </tr>
    <tr>
      <th>70</th>
      <td>product_superfoodssupplements</td>
      <td>-0.010691</td>
    </tr>
    <tr>
      <th>8</th>
      <td>people_ex_baby</td>
      <td>-0.009375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>order_id</td>
      <td>0.008962</td>
    </tr>
    <tr>
      <th>19</th>
      <td>product_babyfood12months</td>
      <td>-0.007080</td>
    </tr>
    <tr>
      <th>16</th>
      <td>day_of_week</td>
      <td>0.006717</td>
    </tr>
    <tr>
      <th>48</th>
      <td>product_haircare</td>
      <td>-0.000009</td>
    </tr>
    <tr>
      <th>6</th>
      <td>global_popularity</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_65_0.png)
    



```python
precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)

desired_recall = 0.35
closest_recall_idx = np.argmin(np.abs(recall - desired_recall))

# Get the threshold corresponding to that index
threshold_at_desired_recall = thresholds[closest_recall_idx]
print(f"Threshold for Recall = {desired_recall}: {threshold_at_desired_recall}")
```

    Threshold for Recall = 0.35: 0.6916317393298211



```python
custom_threshold = 0.69

# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]
y_val_custom_threshold = (y_val_probs > custom_threshold).astype(int)


print("Classification Report:")
print(classification_report(y_val, y_val_custom_threshold))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_custom_threshold))
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.97      0.98    213182
               1       0.14      0.35      0.20      3213
    
        accuracy                           0.96    216395
       macro avg       0.56      0.66      0.59    216395
    weighted avg       0.98      0.96      0.97    216395
    
    Confusion Matrix:
    [[206210   6972]
     [  2083   1130]]


Pretty much same results as before. Lasso took 2h to run & didn't reduced any additonal coeff to 0, obtaining very similar results. This might mean that there is not any irrelevant features among the selected.

Let's try then to play with alpha parameter for stronger regularizations. If alpha increses, then the regularization is stronger. This parameter is adjusted by the C parameter in scikit, which is the inverse of alpha (i.e. higher alpha == lower C)


```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

model1 = LogisticRegression(C=0.5, class_weight='balanced')
model2 = LogisticRegression(C=0.1, class_weight='balanced')
model3 = LogisticRegression(C=0.01, class_weight='balanced')

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
```


```python
# Models with different values of C
models = [model1, model2, model3]
colors = ['r', 'g', 'b']

plt.figure(figsize=(10, 5))

# For each model, calculate and plot ROC and Precision-Recall curves
for i, model in enumerate(models):
    # Calculate probabilities for the positive class
    y_val_probs = model.predict_proba(X_val)[:, 1]

    # ROC Curve
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
    roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

    plt.subplot(1, 2, 1)
    plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})', color=colors[i])

    # Precision-Recall Curve
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

    plt.subplot(1, 2, 2)
    plt.plot(recall_val, precision_val, color=colors[i])

plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()

```


    
![png](push_exploration_files/push_exploration_71_0.png)
    



```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

# Models with different values of C
models = [model1, model2, model3]

for i, model in enumerate(models):
    # Predict the outcomes for the validation data
    y_val_pred = model.predict(X_val)

    # Print the classification report for validation results
    print(f"Classification Report for Model {i + 1}:")
    print(classification_report(y_val, y_val_pred))

    print(f"Confusion Matrix for Model {i + 1}:")
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)

    # Extract feature coefficients
    coefficients = model.coef_[0]  # Coefficients for the first (and only) class
    coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

    # Sort coefficients by absolute value
    coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)

    print(f"Top 10 Features for Model {i + 1}:")
    print(coefficients_df.head(10))

```

    Classification Report for Model 1:
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87    213182
               1       0.04      0.61      0.08      3213
    
        accuracy                           0.78    216395
       macro avg       0.52      0.70      0.48    216395
    weighted avg       0.98      0.78      0.86    216395
    
    Confusion Matrix for Model 1:
    [[166533  46649]
     [  1244   1969]]
    Top 10 Features for Model 1:
                                 Feature  Coefficient
    14                              year    -0.671855
    12      avg_days_to_buy_product_type    -0.606313
    15                             month    -0.525428
    13                         any_event     0.516626
    61                   product_nappies    -0.393317
    4                   normalised_price    -0.296625
    23  product_bathroomlimescalecleaner     0.285124
    38       product_dishwasherdetergent     0.263621
    21           product_babymilkformula    -0.262714
    72         product_tinspackagedfoods     0.248976
    Classification Report for Model 2:
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87    213182
               1       0.04      0.61      0.08      3213
    
        accuracy                           0.78    216395
       macro avg       0.52      0.70      0.48    216395
    weighted avg       0.98      0.78      0.86    216395
    
    Confusion Matrix for Model 2:
    [[166540  46642]
     [  1244   1969]]
    Top 10 Features for Model 2:
                                 Feature  Coefficient
    14                              year    -0.670030
    12      avg_days_to_buy_product_type    -0.598951
    15                             month    -0.523677
    13                         any_event     0.516621
    61                   product_nappies    -0.392399
    4                   normalised_price    -0.285732
    23  product_bathroomlimescalecleaner     0.283179
    38       product_dishwasherdetergent     0.261816
    21           product_babymilkformula    -0.261379
    72         product_tinspackagedfoods     0.248593
    Classification Report for Model 3:
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87    213182
               1       0.04      0.61      0.08      3213
    
        accuracy                           0.78    216395
       macro avg       0.52      0.70      0.48    216395
    weighted avg       0.98      0.78      0.86    216395
    
    Confusion Matrix for Model 3:
    [[166611  46571]
     [  1245   1968]]
    Top 10 Features for Model 3:
                                 Feature  Coefficient
    14                              year    -0.647577
    12      avg_days_to_buy_product_type    -0.537727
    13                         any_event     0.516528
    15                             month    -0.502097
    61                   product_nappies    -0.385560
    23  product_bathroomlimescalecleaner     0.266904
    21           product_babymilkformula    -0.248342
    38       product_dishwasherdetergent     0.246775
    72         product_tinspackagedfoods     0.245437
    60                 product_maternity    -0.218944


Almost same results for different C parameters in l2 regularization.


```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

model = LogisticRegression(C=0.001, class_weight='balanced')

model.fit(X_train, y_train)
```


```python
# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_75_0.png)
    



```python
X = final_data.drop(columns='outcome')
y_val_pred = model.predict(X_val)

# Print the classification report for validation results
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

coefficients = model.coef_[0]  # Coefficients for the first (and only) class
coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)
coefficients_df.tail(10)
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.88    213182
               1       0.04      0.61      0.08      3213
    
        accuracy                           0.78    216395
       macro avg       0.52      0.70      0.48    216395
    weighted avg       0.98      0.78      0.86    216395
    
    Confusion Matrix:
    [[167075  46107]
     [  1247   1966]]





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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43</th>
      <td>product_facialskincare</td>
      <td>0.018726</td>
    </tr>
    <tr>
      <th>54</th>
      <td>product_juicesquash</td>
      <td>0.016255</td>
    </tr>
    <tr>
      <th>7</th>
      <td>count_pets</td>
      <td>0.015768</td>
    </tr>
    <tr>
      <th>16</th>
      <td>day_of_week</td>
      <td>0.009763</td>
    </tr>
    <tr>
      <th>8</th>
      <td>people_ex_baby</td>
      <td>-0.009313</td>
    </tr>
    <tr>
      <th>19</th>
      <td>product_babyfood12months</td>
      <td>-0.003423</td>
    </tr>
    <tr>
      <th>1</th>
      <td>order_id</td>
      <td>-0.002904</td>
    </tr>
    <tr>
      <th>52</th>
      <td>product_householdsundries</td>
      <td>-0.002543</td>
    </tr>
    <tr>
      <th>45</th>
      <td>product_femininecare</td>
      <td>-0.001950</td>
    </tr>
    <tr>
      <th>6</th>
      <td>global_popularity</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Reiterate on feature selection

Based on the best model we achieved, which is using Logistic Regression with l2 and class_weight balanced, we will take at the final weights used (most and least) and do another round of feature selection


```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Initialize model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced')

# Train
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
```


```python
X = final_data.drop(columns='outcome')
coefficients = model.coef_[0]  # Coefficients for the first (and only) class
coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)
coefficients_df.head(10)
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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>year</td>
      <td>-0.672157</td>
    </tr>
    <tr>
      <th>12</th>
      <td>avg_days_to_buy_product_type</td>
      <td>-0.607223</td>
    </tr>
    <tr>
      <th>15</th>
      <td>month</td>
      <td>-0.525724</td>
    </tr>
    <tr>
      <th>13</th>
      <td>any_event</td>
      <td>0.516628</td>
    </tr>
    <tr>
      <th>61</th>
      <td>product_nappies</td>
      <td>-0.393418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>normalised_price</td>
      <td>-0.297199</td>
    </tr>
    <tr>
      <th>23</th>
      <td>product_bathroomlimescalecleaner</td>
      <td>0.285365</td>
    </tr>
    <tr>
      <th>38</th>
      <td>product_dishwasherdetergent</td>
      <td>0.263845</td>
    </tr>
    <tr>
      <th>21</th>
      <td>product_babymilkformula</td>
      <td>-0.262905</td>
    </tr>
    <tr>
      <th>72</th>
      <td>product_tinspackagedfoods</td>
      <td>0.249028</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefficients_df.tail(10)
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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>user_id</td>
      <td>-0.017418</td>
    </tr>
    <tr>
      <th>7</th>
      <td>count_pets</td>
      <td>0.016740</td>
    </tr>
    <tr>
      <th>44</th>
      <td>product_feedingweaning</td>
      <td>-0.014295</td>
    </tr>
    <tr>
      <th>19</th>
      <td>product_babyfood12months</td>
      <td>0.009898</td>
    </tr>
    <tr>
      <th>8</th>
      <td>people_ex_baby</td>
      <td>-0.009382</td>
    </tr>
    <tr>
      <th>1</th>
      <td>order_id</td>
      <td>0.008983</td>
    </tr>
    <tr>
      <th>16</th>
      <td>day_of_week</td>
      <td>0.006718</td>
    </tr>
    <tr>
      <th>70</th>
      <td>product_superfoodssupplements</td>
      <td>-0.002507</td>
    </tr>
    <tr>
      <th>67</th>
      <td>product_shavinggrooming</td>
      <td>-0.000317</td>
    </tr>
    <tr>
      <th>6</th>
      <td>global_popularity</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



- We see any event account for 0.5 in the model (remember we created this variable). We could undo it and introduce the full binary cols, as they seem very relevant
- We see even people_ex_baby was not used, despite we kept the most representative one. We could delete it
- We see day_of_week throwed no info, we could instead present day_of_month
- We see global_popularity was forced to 0, so we can delete it also


```python
binary = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']

cols = information + numerical + categorical + binary + predicted
final_data = filtered_data[cols]

#Verify created at has pandas date_format
final_data['created_at'] = pd.to_datetime(final_data['created_at'])

# Extract year, month, day, and hour as separate features
final_data['year'] = final_data['created_at'].dt.year
final_data['month'] = final_data['created_at'].dt.month
# Extract the day of the week (numerical, 1-7, starting with Monday as 1)
final_data['day_of_month'] = final_data['created_at'].dt.day
final_data['hour'] = final_data['created_at'].dt.hour
final_data = final_data.drop(columns=['created_at'])

# Apply one-hot encoding for 'vendor'
# Drop_first = True to get rid of an additional col. Binaries for keeping consistency as any_event
final_data = pd.get_dummies(final_data, columns=['product_type'], prefix='product', drop_first=True).astype(int)

#Drop cols
final_data = final_data.drop(columns=['global_popularity', 'people_ex_baby'])

print(final_data.shape)
final_data.head()
```

    /tmp/ipykernel_481/2320102903.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['created_at'] = pd.to_datetime(final_data['created_at'])
    /tmp/ipykernel_481/2320102903.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['year'] = final_data['created_at'].dt.year
    /tmp/ipykernel_481/2320102903.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['month'] = final_data['created_at'].dt.month
    /tmp/ipykernel_481/2320102903.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['day_of_month'] = final_data['created_at'].dt.day
    /tmp/ipykernel_481/2320102903.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      final_data['hour'] = final_data['created_at'].dt.hour


    (2163953, 81)





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
      <th>variant_id</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>user_order_seq</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>count_pets</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>outcome</th>
      <th>year</th>
      <th>month</th>
      <th>day_of_month</th>
      <th>hour</th>
      <th>product_allpurposecleaner</th>
      <th>product_babyfood12months</th>
      <th>product_babyfood6months</th>
      <th>product_babymilkformula</th>
      <th>product_babytoiletries</th>
      <th>product_bathroomlimescalecleaner</th>
      <th>product_bathshowergel</th>
      <th>product_beer</th>
      <th>product_binbags</th>
      <th>product_bodyskincare</th>
      <th>product_catfood</th>
      <th>product_cereal</th>
      <th>product_cleaningaccessories</th>
      <th>product_coffee</th>
      <th>product_condimentsdressings</th>
      <th>product_cookingingredientsoils</th>
      <th>product_cookingsaucesmarinades</th>
      <th>product_delicatesstainremover</th>
      <th>product_dental</th>
      <th>product_deodorant</th>
      <th>product_dishwasherdetergent</th>
      <th>product_dogfood</th>
      <th>product_driedfruitsnutsseeds</th>
      <th>product_dryingironing</th>
      <th>product_fabricconditionerfreshener</th>
      <th>product_facialskincare</th>
      <th>product_feedingweaning</th>
      <th>product_femininecare</th>
      <th>product_floorcleanerpolish</th>
      <th>product_foodstorage</th>
      <th>product_haircare</th>
      <th>product_handsoapsanitisers</th>
      <th>product_healthcarevitamins</th>
      <th>product_homebaking</th>
      <th>product_householdsundries</th>
      <th>product_jamhoneyspreads</th>
      <th>product_juicesquash</th>
      <th>product_kidsdental</th>
      <th>product_kidssnacks</th>
      <th>product_kitchenovencleaner</th>
      <th>product_kitchenrolltissues</th>
      <th>product_longlifemilksubstitutes</th>
      <th>product_maternity</th>
      <th>product_nappies</th>
      <th>product_nappypants</th>
      <th>product_petcare</th>
      <th>product_pickledfoodolives</th>
      <th>product_premixedcocktails</th>
      <th>product_ricepastapulses</th>
      <th>product_shavinggrooming</th>
      <th>product_snacksconfectionery</th>
      <th>product_softdrinksmixers</th>
      <th>product_superfoodssupplements</th>
      <th>product_tea</th>
      <th>product_tinspackagedfoods</th>
      <th>product_toiletroll</th>
      <th>product_washingcapsules</th>
      <th>product_washingliquidgel</th>
      <th>product_washingpowder</th>
      <th>product_windowglasscleaner</th>
      <th>product_wipescottonwool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>5</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>33</td>
      <td>42</td>
      <td>30</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2020</td>
      <td>10</td>
      <td>6</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Train again the model


```python
X = final_data.drop(columns='outcome')
y = final_data['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split again
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=33)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=33)

# Initialize model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced')

# Train
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
```


```python
X = final_data.drop(columns='outcome')
y_val_pred = model.predict(X_val)

# Print the classification report for validation results
print("Classification Report:")
print(classification_report(y_val, y_val_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)

coefficients = model.coef_[0]  # Coefficients for the first (and only) class
coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)
coefficients_df.head(10)
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.78      0.87    213182
               1       0.04      0.62      0.08      3213
    
        accuracy                           0.77    216395
       macro avg       0.52      0.70      0.47    216395
    weighted avg       0.98      0.77      0.86    216395
    
    Confusion Matrix:
    [[165490  47692]
     [  1229   1984]]





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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>year</td>
      <td>-0.747774</td>
    </tr>
    <tr>
      <th>10</th>
      <td>avg_days_to_buy_product_type</td>
      <td>-0.607219</td>
    </tr>
    <tr>
      <th>16</th>
      <td>month</td>
      <td>-0.593842</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ordered_before</td>
      <td>0.434268</td>
    </tr>
    <tr>
      <th>62</th>
      <td>product_nappies</td>
      <td>-0.395642</td>
    </tr>
    <tr>
      <th>24</th>
      <td>product_bathroomlimescalecleaner</td>
      <td>0.285216</td>
    </tr>
    <tr>
      <th>22</th>
      <td>product_babymilkformula</td>
      <td>-0.270324</td>
    </tr>
    <tr>
      <th>39</th>
      <td>product_dishwasherdetergent</td>
      <td>0.263882</td>
    </tr>
    <tr>
      <th>73</th>
      <td>product_tinspackagedfoods</td>
      <td>0.249627</td>
    </tr>
    <tr>
      <th>31</th>
      <td>product_cleaningaccessories</td>
      <td>0.235490</td>
    </tr>
  </tbody>
</table>
</div>




```python
coefficients_df.tail(15)
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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>product_kidsdental</td>
      <td>-0.035290</td>
    </tr>
    <tr>
      <th>13</th>
      <td>active_snoozed</td>
      <td>0.035236</td>
    </tr>
    <tr>
      <th>7</th>
      <td>days_since_purchase_variant_id</td>
      <td>-0.034834</td>
    </tr>
    <tr>
      <th>53</th>
      <td>product_householdsundries</td>
      <td>0.024792</td>
    </tr>
    <tr>
      <th>28</th>
      <td>product_bodyskincare</td>
      <td>0.024669</td>
    </tr>
    <tr>
      <th>55</th>
      <td>product_juicesquash</td>
      <td>0.022988</td>
    </tr>
    <tr>
      <th>18</th>
      <td>hour</td>
      <td>0.020846</td>
    </tr>
    <tr>
      <th>45</th>
      <td>product_feedingweaning</td>
      <td>-0.013937</td>
    </tr>
    <tr>
      <th>6</th>
      <td>count_pets</td>
      <td>0.012153</td>
    </tr>
    <tr>
      <th>1</th>
      <td>order_id</td>
      <td>0.011454</td>
    </tr>
    <tr>
      <th>2</th>
      <td>user_id</td>
      <td>-0.010594</td>
    </tr>
    <tr>
      <th>20</th>
      <td>product_babyfood12months</td>
      <td>0.010576</td>
    </tr>
    <tr>
      <th>17</th>
      <td>day_of_month</td>
      <td>-0.009731</td>
    </tr>
    <tr>
      <th>68</th>
      <td>product_shavinggrooming</td>
      <td>-0.002280</td>
    </tr>
    <tr>
      <th>71</th>
      <td>product_superfoodssupplements</td>
      <td>-0.001916</td>
    </tr>
  </tbody>
</table>
</div>



Model behaviour it's pretty similar for P = 0.5 in the confusion matrix. Let's try to plot the curves, specially the precision-recall again


```python
# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]

# ROC Curve
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_probs)
roc_auc_val = auc(fpr_val, tpr_val)  # Calculate AUC

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, label=f'ROC (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# Precision-Recall Curve
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_probs)

plt.subplot(1, 2, 2)
plt.plot(recall_val, precision_val)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
```


    
![png](push_exploration_files/push_exploration_89_0.png)
    


ROC is similar but we see a different pattern in the precision vs recall behaviour. Let's explote them using different threshold for maximising precison while we get a sufficient recall to appreciate positive values. We see that for same recall as we selected before (0.32 approx) we obtain similar or even worse results. However, if we select a little fewer recall, we obtain significant increases in precision vs previous model. Let's use 0.3, 0.2, and 0.15.


```python
recall
```


```python
[5.15846135e-05 6.75884443e-05 7.46107959e-05 ... 9.99949782e-01
 9.99958761e-01 9.99979388e-01]
```


```python
precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)

desired_recalls = [0.3, 0.2, 0.15]
thresholds_at_desired_recalls = {}

for i in desired_recalls:
    closest_recall_idx = np.argmin(np.abs(recall - i))
    
    # Get the threshold corresponding to that index
    threshold_at_desired_recall = thresholds[closest_recall_idx]
    
    thresholds_at_desired_recalls[i] = threshold_at_desired_recall

for recall_value, threshold_value in thresholds_at_desired_recalls.items():
    print(f"Threshold for Recall = {recall_value}: {threshold_value}")

```

    Threshold for Recall = 0.3: 0.8495145097324593
    Threshold for Recall = 0.2: 0.9326935056189675
    Threshold for Recall = 0.15: 0.9510581366637918



```python
thresholds_at_desired_recalls.items
```




    <function dict.items>




```python
y_val_probs = model.predict_proba(X_val)[:, 1]

for recall, threshold in thresholds_at_desired_recalls.items():
    # Calculate probabilities for the positive class
    y_val_custom_threshold = (y_val_probs > threshold).astype(int)
    print(f"Threshold: {threshold}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_custom_threshold))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_custom_threshold),"\n")
```

    Threshold: 0.8495145097324593
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.98      0.98    213182
               1       0.19      0.30      0.23      3213
    
        accuracy                           0.97    216395
       macro avg       0.59      0.64      0.61    216395
    weighted avg       0.98      0.97      0.97    216395
    
    Confusion Matrix:
    [[208946   4236]
     [  2249    964]] 
    
    Threshold: 0.9326935056189675
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99    213182
               1       0.23      0.20      0.21      3213
    
        accuracy                           0.98    216395
       macro avg       0.61      0.60      0.60    216395
    weighted avg       0.98      0.98      0.98    216395
    
    Confusion Matrix:
    [[211054   2128]
     [  2570    643]] 
    
    Threshold: 0.9510581366637918
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99    213182
               1       0.28      0.15      0.20      3213
    
        accuracy                           0.98    216395
       macro avg       0.63      0.57      0.59    216395
    weighted avg       0.98      0.98      0.98    216395
    
    Confusion Matrix:
    [[211946   1236]
     [  2731    482]] 
    


Good behaviour for adjusting the precision of positives, but we are losing slightly more positives that we're not catching. This model seems to work better with lower recall. Let's try to validate 0.22, 0.19 and 0.17.


```python
precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)

desired_recalls = [0.25, 0.22, 0.19]
thresholds_at_desired_recalls = {}

for i in desired_recalls:
    closest_recall_idx = np.argmin(np.abs(recall - i))
    
    # Get the threshold corresponding to that index
    threshold_at_desired_recall = thresholds[closest_recall_idx]
    
    thresholds_at_desired_recalls[i] = threshold_at_desired_recall

for recall_value, threshold_value in thresholds_at_desired_recalls.items():
    print(f"Threshold for Recall = {recall_value}: {threshold_value}")

```

    Threshold for Recall = 0.25: 0.9080735787919318
    Threshold for Recall = 0.22: 0.9249901789056006
    Threshold for Recall = 0.19: 0.937330083213395



```python
y_val_probs = model.predict_proba(X_val)[:, 1]

for recall, threshold in thresholds_at_desired_recalls.items():
    # Calculate probabilities for the positive class
    y_val_custom_threshold = (y_val_probs > threshold).astype(int)
    print(f"Threshold: {threshold}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_custom_threshold))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_custom_threshold),"\n")
```

    Threshold: 0.9080735787919318
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99    213182
               1       0.20      0.25      0.22      3213
    
        accuracy                           0.97    216395
       macro avg       0.59      0.62      0.60    216395
    weighted avg       0.98      0.97      0.98    216395
    
    Confusion Matrix:
    [[209995   3187]
     [  2410    803]] 
    
    Threshold: 0.9249901789056006
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99    213182
               1       0.22      0.22      0.22      3213
    
        accuracy                           0.98    216395
       macro avg       0.60      0.60      0.60    216395
    weighted avg       0.98      0.98      0.98    216395
    
    Confusion Matrix:
    [[210672   2510]
     [  2506    707]] 
    
    Threshold: 0.937330083213395
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99    213182
               1       0.24      0.19      0.21      3213
    
        accuracy                           0.98    216395
       macro avg       0.62      0.59      0.60    216395
    weighted avg       0.98      0.98      0.98    216395
    
    Confusion Matrix:
    [[211291   1891]
     [  2603    610]] 
    


FP / TP is better. A 1/4 of the alerts sent will be bought. So it we would like to opt for a more conservative model, this might be the choice. Additionally, if we want to be really sure about the predictions for positives, we can set the recall to 0.1:


```python
precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)

desired_recalls = [0.1]
thresholds_at_desired_recalls = {}

for i in desired_recalls:
    closest_recall_idx = np.argmin(np.abs(recall - i))
    
    # Get the threshold corresponding to that index
    threshold_at_desired_recall = thresholds[closest_recall_idx]
    
    thresholds_at_desired_recalls[i] = threshold_at_desired_recall

for recall_value, threshold_value in thresholds_at_desired_recalls.items():
    print(f"Threshold for Recall = {recall_value}: {threshold_value}")

```

    Threshold for Recall = 0.1: 0.9726607989659661



```python
y_val_probs = model.predict_proba(X_val)[:, 1]

for recall, threshold in thresholds_at_desired_recalls.items():
    # Calculate probabilities for the positive class
    y_val_custom_threshold = (y_val_probs > threshold).astype(int)
    print(f"Threshold: {threshold}")
    print("Classification Report:")
    print(classification_report(y_val, y_val_custom_threshold))

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_custom_threshold),"\n")
```

    Threshold: 0.9726607989659661
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      1.00      0.99    213182
               1       0.35      0.10      0.16      3213
    
        accuracy                           0.98    216395
       macro avg       0.67      0.55      0.57    216395
    weighted avg       0.98      0.98      0.98    216395
    
    Confusion Matrix:
    [[212597    585]
     [  2892    321]] 
    


## Assesing models vs. target impact

The targeted impact for the problem was to increase monthly sales by 2% and a boost of 25% over selected items. Let's do a few hypothesis and then some quick numbers:

- We assume monthly volume as the mean for the last 3 months provided in the data, so JFM 21. We get that number very generalistic from previous module, roughly let's say it was £20k per month. So we expect an increase of £200/m
- If we expect a boost of 25% over selected items, let's say that we will be adding +25% units for all the items that we selected for a notification.

If we assume that the avg value for all the items can be estimated dividing the monthly value (20k) by the total ordered items by month from that period (~4000), we get it is around £5. So we will need to send a total of 200/5 = 40 items bought exclusively because of the push notifications. 

From there, we can opt for the more conservative model as we will be meeting completely the expectations and doublechecking with very high confident, that we do not spam the user too much. If we apply these number for the last model we try with recall = 0.19:


```python
precision, recall, thresholds = precision_recall_curve(y_val, y_val_probs)

desired_recall = 0.19
closest_recall_idx = np.argmin(np.abs(recall - desired_recall))

# Get the threshold corresponding to that index
threshold_at_desired_recall = thresholds[closest_recall_idx]
print(f"Threshold for Recall = {desired_recall}: {threshold_at_desired_recall}")
```

    Threshold for Recall = 0.19: 0.937330083213395



```python
custom_threshold = 0.937

# Calculate probabilities for the positive class
y_val_probs = model.predict_proba(X_val)[:, 1]
y_val_custom_threshold = (y_val_probs > custom_threshold).astype(int)


print("Classification Report:")
print(classification_report(y_val, y_val_custom_threshold))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_val_custom_threshold))
```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99    213182
               1       0.24      0.19      0.21      3213
    
        accuracy                           0.98    216395
       macro avg       0.62      0.59      0.60    216395
    weighted avg       0.98      0.98      0.98    216395
    
    Confusion Matrix:
    [[211278   1904]
     [  2600    613]]



```python
filtered_data['variant_id'].nunique()
```




    913



We will be sending twice a month: e.g. 2.5k push notification, generating 600 items bought for sure. From there, we can say our open rate will be 25% at very minimun, with a more realistic approach about 30-50% (assuming a conversion rate per notification opened ~50%).

Now let's penalize that the items that we're predicting will be bought whether the notifications are sent or not, by a factor of 50% less impact of the notifications.

So from the example, we will be sending 5k notifications/month with a total impact estimated of 600 items extra sold, resulting in £3k (+15% uplift). The products selected for the promo we can assume will be mainly new promos/discounts, so the target will be more than surprased by this increase.


# 2. Second and improved approach

Previous version was a brute and very forced model, trying with too many features since the beginning and no baseline defined. Here we will take the lessons learned and apply the following:

1. Define the minimun features as a first approach and then, if needed, add more.
2. Proper train/validation split taking into account the temporality of data and avoiding information leakage.
3. Defining a baseline as a our worst estimator for comparing the model
4. Play with different parameters dynamically when training models
5. Final decision

## 1) Features


```python
filtered_data.columns
```




    Index(['variant_id', 'product_type', 'order_id', 'user_id', 'created_at',
           'order_date', 'user_order_seq', 'outcome', 'ordered_before',
           'abandoned_before', 'active_snoozed', 'set_as_regular',
           'normalised_price', 'discount_pct', 'vendor', 'global_popularity',
           'count_adults', 'count_children', 'count_babies', 'count_pets',
           'people_ex_baby', 'days_since_purchase_variant_id',
           'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
           'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
           'std_days_to_buy_product_type'],
          dtype='object')




```python
# Initial classification:

predicted = ['outcome']
information = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
numerical = ['user_order_seq', 'normalised_price', 'discount_pct', 'global_popularity',
            'count_adults', 'count_children', 'count_babies', 'count_pets', 
            'people_ex_baby', 'days_since_purchase_variant_id', 
            'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
            'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
                'std_days_to_buy_product_type']

categorical = ['product_type', 'vendor']
binary = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
```

## 2) Train/validation/test set

Makes sense to apply temporal splitting since: i) it is an increasing temporal series ii) we might incur in information leakage if we split several orders across train/test. We could make the point that the feature created_at would incorporate that info into the prediction, but it is also true that the data has changed too much since beginning that it is not justified no just incorporate that variable as a separate one and continue with random samples.

There could be different ways of applyinh this (sliding window validation if we want to test in different points in time or a walk-forward validation splitting). In this case, let's select a simple walk-forward with first 4 months for training, 2 weeks after for validation and 2 last weeks for testing.


```python
filtered_data['order_date'] = pd.to_datetime(filtered_data['order_date']).dt.date
daily_orders = filtered_data.groupby('order_date').order_id.nunique()
plt.plot(daily_orders)
```

    /tmp/ipykernel_462/3141760495.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      filtered_data['order_date'] = pd.to_datetime(filtered_data['order_date']).dt.date





    [<matplotlib.lines.Line2D at 0x7f0d54ef6210>]




    
![png](push_exploration_files/push_exploration_116_2.png)
    



```python
print(daily_orders.index.max())
print(daily_orders.index.min())
```

    2021-03-03
    2020-10-05



```python
num_days = len(daily_orders)

test_days = 14
validation_days = 14
train_days = num_days - test_days - validation_days

# Train
train_start = daily_orders.index.min()
train_end = train_start + timedelta(days = train_days)

# Validation
validation_end = train_end + timedelta(days = validation_days)

# Test
test_end = daily_orders.index.max()
```


```python
train = filtered_data[filtered_data.order_date <= train_end]
val = filtered_data[(filtered_data.order_date > train_end) & (filtered_data.order_date <= validation_end)]
test = filtered_data[filtered_data.order_date > validation_end]

print(f"Train set: {len(train) / len(filtered_data):.2f}")
print(f"Train set: {len(val) / len(filtered_data):.2f}")
print(f"Train set: {len(test) / len(filtered_data):.2f}")
```

    Train set: 0.65
    Train set: 0.16
    Train set: 0.19



```python
# Let's give a lit bit extra days for train
test_days = 10
validation_days = 10
train_days = num_days - test_days - validation_days

# Train
train_start = daily_orders.index.min()
train_end = train_start + timedelta(days = train_days)

validation_end = train_end + timedelta(days = validation_days)

test_end = daily_orders.index.max()

train = filtered_data[filtered_data.order_date <= train_end]
val = filtered_data[(filtered_data.order_date > train_end) & (filtered_data.order_date <= validation_end)]
test = filtered_data[filtered_data.order_date > validation_end]

print(f"Train set: {len(train) / len(filtered_data):.2f}")
print(f"Train set: {len(val) / len(filtered_data):.2f}")
print(f"Train set: {len(test) / len(filtered_data):.2f}")

```

    Train set: 0.73
    Train set: 0.13
    Train set: 0.13



```python
#Let's define the curves into one just func

def get_roc_pr_curves(model_name, y_pred, y_test):

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})-{model_name}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f})-{model_name}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")


    plt.tight_layout()
    plt.show()
```



## 3) Baseline

Let's assess the model against some basic benchmark as a proxy to verify that our model is, at least, equal to our best estimator available. For that we could take into account the most purchased products/predict the week before/global popularity or ordered_before. Let's use the the ordered_before and also the 10 days before using just numerical features.


```python
# Ordered before
feature = ['ordered_before']
predicted = ['outcome']
name = 'Ordered'
get_roc_pr_curves(model_name=name y_pred=val[feature], y_test=val[predicted])

# Predict the week before
train_start = train_end - timedelta(days = 10)
train = filtered_data[(filtered_data.order_date > train_start) & (filtered_data.order_date <= train_end)]
val = filtered_data[(filtered_data.order_date > train_end) & (filtered_data.order_date <= validation_end)]

cols = numerical 
train = train[cols+predicted]
val = val[cols+predicted]

X = train[cols]
y = train['outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)
lr = LogisticRegression()
model.fit(X, y)

x_val_pred = val.drop(columns=predicted)
y_val_pred = model.predict(x_val_pred)

get_roc_pr_curves(model_name=lr, y_pred=y_val_pred, y_test=val[predicted])
```


    
![png](push_exploration_files/push_exploration_125_0.png)
    


    /home/dan1dr/.cache/pypoetry/virtualenvs/zrive-ds-UEx3J_CK-py3.11/lib/python3.11/site-packages/sklearn/base.py:458: UserWarning: X has feature names, but LogisticRegression was fitted without feature names
      warnings.warn(



    
![png](push_exploration_files/push_exploration_125_2.png)
    


'ordered_before' works would be our baseline (slightly better than a random classifier while second approach behaves like that); anyway, let's move ahead and start from simple to more complex models

## Train model


```python
# Let's continue with the previous splitting

num_days = len(daily_orders)

test_days = 10
validation_days = 10
train_days = num_days - test_days - validation_days

# Train
train_start = daily_orders.index.min()
train_end = train_start + timedelta(days = train_days)

validation_end = train_end + timedelta(days = validation_days)

test_end = daily_orders.index.max()

# We will use in this first case numerical + binary cols
cols = numerical + binary + predicted

train = filtered_data[filtered_data.order_date <= train_end][cols]
val = filtered_data[(filtered_data.order_date > train_end) & (filtered_data.order_date <= validation_end)][cols]
test = filtered_data[filtered_data.order_date > validation_end][cols]

print(f"Train set: {len(train) / len(filtered_data):.2f}")
print(f"Train set: {len(val) / len(filtered_data):.2f}")
print(f"Train set: {len(test) / len(filtered_data):.2f}")
```

    Train set: 0.73
    Train set: 0.13
    Train set: 0.13



```python
# Let's define a function to automtically split the datasets

def split_sets(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    '''
    Return a df with X (features) and a series y (outcome) set'''
    X = df.drop(columns=label)
    y = df[label]
    return X, y.stack()

# Weird that I need to use .stack() to return a Series despite using -> pd.Series
```


```python
X_train, y_train = split_sets(train, predicted)
X_val, y_val = split_sets(val, predicted)
X_test, y_test = split_sets(test, predicted)
```


```python
X_train.columns
```




    Index(['user_order_seq', 'normalised_price', 'discount_pct',
           'global_popularity', 'count_adults', 'count_children', 'count_babies',
           'count_pets', 'people_ex_baby', 'days_since_purchase_variant_id',
           'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
           'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
           'std_days_to_buy_product_type', 'ordered_before', 'abandoned_before',
           'active_snoozed', 'set_as_regular'],
          dtype='object')




```python
from sklearn.pipeline import Pipeline

# Define a parameter grid for C values
param_grid = {
    'model__C': np.logspace(-4, 4, 9)  # You can adjust the range as needed
}

# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l2', solver='liblinear'))
])

# Create a GridSearchCV instance to find the best C parameter
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Fit the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model using ROC and Precision-Recall curves
y_val_pred = best_model.predict_proba(X_val)[:, 1]  # Predict probabilities for positive class

# Get the best C parameter from the grid search
best_C = best_model.named_steps['model'].C

# Display the best C parameter
print(f"Best C parameter: {best_C}")

# Plot ROC and Precision-Recall curves
get_roc_pr_curves(model_name=f"Logistic Regression (C={best_C})", y_pred=y_val_pred, y_test=y_val)

# Show the ROC and Precision-Recall curves
plt.show()
```

    Best C parameter: 0.0001



    
![png](push_exploration_files/push_exploration_132_1.png)
    



```python
best_model = grid_search.best_estimator_
# With named_steps we access specific components for the pipe created
coefficients = best_model.named_steps['model'].coef_[0]  # Coefficients for the first (and only) class

# Create a DataFrame to display the coefficients along with feature names
coefficients_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coefficients})

# Sort the DataFrame by the absolute value of coefficients in descending order
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)

# Display the top 10 features with the largest absolute coefficients
features_weights = coefficients_df
print(features_weights)
```

                                 Feature  Coefficient
    15                    ordered_before     0.247194
    3                  global_popularity     0.169655
    16                  abandoned_before     0.116936
    1                   normalised_price    -0.063870
    18                    set_as_regular     0.042379
    10        avg_days_to_buy_variant_id    -0.034828
    0                     user_order_seq    -0.034564
    12  days_since_purchase_product_type     0.024327
    13      avg_days_to_buy_product_type    -0.019759
    7                         count_pets     0.009199
    2                       discount_pct     0.008600
    9     days_since_purchase_variant_id    -0.008257
    11        std_days_to_buy_variant_id    -0.005999
    17                    active_snoozed     0.005637
    5                     count_children    -0.002467
    4                       count_adults     0.002293
    6                       count_babies    -0.002091
    8                     people_ex_baby    -0.000693
    14      std_days_to_buy_product_type     0.000112


We see that from std_days_to_buy onwards features are almost not used. We see the improvement of this model and the optimal 'C' found = e-4. Let's try with Lasso:


```python
# Define a parameter grid for C values
param_grid = {
    'model__C': np.logspace(-4, 4, 9)  # You can adjust the range as needed
}

# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='liblinear'))
])

# Create a GridSearchCV instance to find the best C parameter
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Fit the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model using ROC and Precision-Recall curves
y_val_pred = best_model.predict_proba(X_val)[:, 1]  # Predict probabilities for positive class

# Get the best C parameter from the grid search
best_C = best_model.named_steps['model'].C

# Display the best C parameter
print(f"Best C parameter: {best_C}")

# Plot ROC and Precision-Recall curves
get_roc_pr_curves(model_name=f"Logistic Regression (C={best_C})", y_pred=y_val_pred, y_test=y_val)

# Show the ROC and Precision-Recall curves
plt.show()
```

    Best C parameter: 0.0001



    
![png](push_exploration_files/push_exploration_135_1.png)
    


Pretty same results with same best parameter. Let's try to see if any extra coefficients was forced to 0 vs ridge:


```python
coefficients = best_model.named_steps['model'].coef_[0]  # Coefficients for the first (and only) class

coefficients_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coefficients})
coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)
features_weights = coefficients_df
print(features_weights)
```

                                 Feature  Coefficient
    15                    ordered_before     0.291926
    3                  global_popularity     0.161619
    16                  abandoned_before     0.103754
    18                    set_as_regular     0.006035
    10        avg_days_to_buy_variant_id     0.000000
    17                    active_snoozed     0.000000
    14      std_days_to_buy_product_type     0.000000
    13      avg_days_to_buy_product_type     0.000000
    12  days_since_purchase_product_type     0.000000
    11        std_days_to_buy_variant_id     0.000000
    0                     user_order_seq     0.000000
    1                   normalised_price     0.000000
    8                     people_ex_baby     0.000000
    7                         count_pets     0.000000
    6                       count_babies     0.000000
    5                     count_children     0.000000
    4                       count_adults     0.000000
    2                       discount_pct     0.000000
    9     days_since_purchase_variant_id     0.000000


Lasso only used ordered_before, global_popularity, abandoned_before and set_as_regular, obtaining very similar results vs. Ridge. With that, let's try to use same features again but with ridge:


```python
X_train[cols]
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
      <th>ordered_before</th>
      <th>global_popularity</th>
      <th>abandoned_before</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.038462</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.038462</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2879664</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2879665</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2879666</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2879667</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2879668</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1589265 rows × 3 columns</p>
</div>




```python
cols = ['ordered_before', 'global_popularity', 'abandoned_before']

lr_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty="l2", C=1e-4))
])
lr_ridge.fit(X_train[cols], y_train)

y_val_pred = lr_ridge.predict_proba(X_val[cols])[:, 1]
get_roc_pr_curves(model_name=f"Logistic Regression (C={1e-4})", y_pred=y_val_pred, y_test=y_val)


```


    
![png](push_exploration_files/push_exploration_140_0.png)
    



```python
# Let's continue with the previous splitting
test_days = 10
validation_days = 10
train_days = num_days - test_days - validation_days

# Train
train_start = daily_orders.index.min()
train_end = train_start + timedelta(days = train_days)

validation_end = train_end + timedelta(days = validation_days)

test_end = daily_orders.index.max()

# We will use in this first case numerical + binary cols
cols = ['ordered_before', 'global_popularity', 'abandoned_before', 'outcome']
cols.append('product_type')

train = filtered_data[filtered_data.order_date <= train_end][cols]
val = filtered_data[(filtered_data.order_date > train_end) & (filtered_data.order_date <= validation_end)][cols]
test = filtered_data[filtered_data.order_date > validation_end][cols]

print(f"Train set: {len(train) / len(filtered_data):.2f}")
print(f"Train set: {len(val) / len(filtered_data):.2f}")
print(f"Train set: {len(test) / len(filtered_data):.2f}")
```

    Train set: 0.73
    Train set: 0.13
    Train set: 0.13



```python
X_train, y_train = split_sets(train, predicted)
X_val, y_val = split_sets(val, predicted)
X_test, y_test = split_sets(test, predicted)

X_train = pd.get_dummies(X_train, columns=['product_type'], prefix='product', drop_first=True).astype(int)
# We add this drop bc it was not present during training sent
X_val = pd.get_dummies(X_val, columns=['product_type'], prefix='product', drop_first=True).astype(int).drop(columns='product_feedingweaning')
```


```python
lr_ridge = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty="l2", C=1e-4))
])
lr_ridge.fit(X_train, y_train)

y_val_pred = lr_ridge.predict_proba(X_val)[:, 1]
get_roc_pr_curves(model_name=f"Logistic Regression (C={1e-4})", y_pred=y_val_pred, y_test=y_val)
```


    
![png](push_exploration_files/push_exploration_143_0.png)
    


Nah, nothing remarkable. Interesting idea I wanted to validate: if I train a model with n features and Lasso forces m coeff to 0, then training again Lasso with n-m features might throw similar results? -> No, as the features before training changed and, as such, relevant features might change. The diamond (or in this case: Lasso polytope, as features are more than 2D) it's a different one. I tried to visualize the equivalent-ellipse-but-in >3D intersecting with this polytope but no ty.

## Insights

- Lasso and Ridge seem to behave very similarly, yielding slightly better results with stronger regularization.

- Categorical variables offer little to no additional predictive power, so we will select only binary and numerical columns.

- We want to be on the left side of the Precision-Recall curve, aiming to minimize false positives as much as possible. The Precision-Recall curve is given by:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

where TP represents true positives and FP represents false positives.

- The ROC curve is not very relevant as classes are heavily unbalanced. With False Positive Rate (FPR) defined as:

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

and considering that $$TN \gg FP, \quad  FPR \rightarrow 0$$ very quickly, so it is not our best friend for assessing visually the efficacy there.

- We could try different encondings, parameters and regularisations, but for keeping it simple, I will incorporate here the final decision with Ridge and l2 penalty, using the Pipeline functionality with different params.



```python
cols = numerical + binary + predicted

train = filtered_data[filtered_data.order_date <= train_end]
val = filtered_data[(filtered_data.order_date > train_end) & (filtered_data.order_date <= validation_end)]
test = filtered_data[filtered_data.order_date > validation_end]

X_train, y_train = split_sets(train, predicted)
X_val, y_val = split_sets(val, predicted)
X_test, y_test = split_sets(test, predicted)
```


```python
import logging

# Configure logging system
logging.basicConfig(
    level=logging.INFO,  # Set logging level (others: DEBUG, WARNING, ERROR, etc.)
    format="%(levelname)s - %(message)s",  # Define the log message format
)
```


```python
pd.set_option("display.precision", 4)

def assess_NA(data: pd.DataFrame):
    """
    Returns a pd.DataFrame denoting the total number of NA
    values and the percentage of NA values in each column.
    The column names are noted on the index.
    """
    # pd.Datadenoting features and the sum of their null values
    nulls = data.isnull().sum().reset_index().rename(columns = {0: "count"})
    nulls['percent'] = nulls['count']*100/len(data)

    return nulls
```


```python
def read_data(file_path: str) -> pd.DataFrame:
    '''
    Read and validate data from a CSV file.
    Parameters: file_path (str)
    Returns: pd.DataFrame
    '''
    try:
        data = pd.read_csv(file_path)

        if data is not None and not data.empty:
            logging.info("Data loaded successfully.")

            # Info about its shape and nulls
            rows, cols = data.shape
            logging.info(f"Data shape: {rows} rows x {cols} columns")

            null_assessment = assess_NA(data)
            logging.info("Assessment of NA values:\n" + null_assessment.to_string(index=False))

            # Show data sample
            print(data.head(5))
            
            return data
        else:
            logging.error("Error: The loaded data is empty or None")
            return
        
    except FileNotFoundError as e:
        logging.error("Error: The CSV file is empty")
    except pd.errors.EmptyDataError as e:
        logging.error("Error: An unexpected error occurred while loading the data.")
```


```python
data_loaded = read_data(path)
```

    INFO - Data loaded successfully.
    INFO - Data shape: 2880549 rows x 27 columns
    INFO - Assessment of NA values:
                               index  count  percent
                          variant_id      0      0.0
                        product_type      0      0.0
                            order_id      0      0.0
                             user_id      0      0.0
                          created_at      0      0.0
                          order_date      0      0.0
                      user_order_seq      0      0.0
                             outcome      0      0.0
                      ordered_before      0      0.0
                    abandoned_before      0      0.0
                      active_snoozed      0      0.0
                      set_as_regular      0      0.0
                    normalised_price      0      0.0
                        discount_pct      0      0.0
                              vendor      0      0.0
                   global_popularity      0      0.0
                        count_adults      0      0.0
                      count_children      0      0.0
                        count_babies      0      0.0
                          count_pets      0      0.0
                      people_ex_baby      0      0.0
      days_since_purchase_variant_id      0      0.0
          avg_days_to_buy_variant_id      0      0.0
          std_days_to_buy_variant_id      0      0.0
    days_since_purchase_product_type      0      0.0
        avg_days_to_buy_product_type      0      0.0
        std_days_to_buy_product_type      0      0.0


           variant_id     product_type       order_id        user_id  \
    0  33826472919172  ricepastapulses  2807985930372  3482464092292   
    1  33826472919172  ricepastapulses  2808027644036  3466586718340   
    2  33826472919172  ricepastapulses  2808099078276  3481384026244   
    3  33826472919172  ricepastapulses  2808393957508  3291363377284   
    4  33826472919172  ricepastapulses  2808429314180  3537167515780   
    
                created_at           order_date  user_order_seq  outcome  \
    0  2020-10-05 16:46:19  2020-10-05 00:00:00               3      0.0   
    1  2020-10-05 17:59:51  2020-10-05 00:00:00               2      0.0   
    2  2020-10-05 20:08:53  2020-10-05 00:00:00               4      0.0   
    3  2020-10-06 08:57:59  2020-10-06 00:00:00               2      0.0   
    4  2020-10-06 10:37:05  2020-10-06 00:00:00               3      0.0   
    
       ordered_before  abandoned_before  ...  count_children  count_babies  \
    0             0.0               0.0  ...             0.0           0.0   
    1             0.0               0.0  ...             0.0           0.0   
    2             0.0               0.0  ...             0.0           0.0   
    3             0.0               0.0  ...             0.0           0.0   
    4             0.0               0.0  ...             0.0           0.0   
    
       count_pets  people_ex_baby days_since_purchase_variant_id  \
    0         0.0             2.0                           33.0   
    1         0.0             2.0                           33.0   
    2         0.0             2.0                           33.0   
    3         0.0             2.0                           33.0   
    4         0.0             2.0                           33.0   
    
       avg_days_to_buy_variant_id  std_days_to_buy_variant_id  \
    0                        42.0                     31.1341   
    1                        42.0                     31.1341   
    2                        42.0                     31.1341   
    3                        42.0                     31.1341   
    4                        42.0                     31.1341   
    
       days_since_purchase_product_type  avg_days_to_buy_product_type  \
    0                              30.0                          30.0   
    1                              30.0                          30.0   
    2                              30.0                          30.0   
    3                              30.0                          30.0   
    4                              30.0                          30.0   
    
       std_days_to_buy_product_type  
    0                       24.2762  
    1                       24.2762  
    2                       24.2762  
    3                       24.2762  
    4                       24.2762  
    
    [5 rows x 27 columns]



```python
def preprocess_data(data: pd.DataFrame, remove_if_all_na: bool = False, num_items: int = 5) -> pd.DataFrame:
    '''
    Preprocess data by removing rows with 
    filter orders with at least 5 items

    Parameters: 
    - data (pd.DataFrame)
    - remove_all_na_rows (bool): If True, remove rows where at least one value is missing. 
    If False, remove rows where all values are missing.

    Returns: 
    - pd.DataFrame. The preprocessed dataset
    '''
    try:
        initial_length = len(data)
        if remove_if_all_na: # remove if everything is NA
            data = data.dropna(how='all')
        else:
            data = data.dropna()
        dropped_length = len(data)

        # Filter orders with >= 5 items
        num_items_ordered = data.groupby('order_id')['outcome'].transform('sum')
        processed_data = data[num_items_ordered >= num_items]

        logging.info(f"Length initial data: {len(data)}")
        logging.info(f"Rows dropped with NA's: {initial_length - dropped_length}")
        logging.info(f"Length filtered data: {len(processed_data)}\n")
 
        return processed_data
    
    except FileNotFoundError as e:
        logging.error("Error: File not found.")
```


```python
preprocessed_data = preprocess_data(data_loaded)
```

    INFO - Length initial data: 2880549
    INFO - Rows dropped with NA's: 0
    INFO - Length filtered data: 2163953
    



```python
def split_sets(df: pd.DataFrame, label: str) -> (pd.DataFrame, pd.Series):
    '''
    Return a df with X set (features) and a series y set (outcome)

    Parameters:
    - df: pd.DataFrame

    Returns:
    - X: pd.DataFrame (features only)
    - y: pd.Series (label to predict)
    '''
    X = df.drop(columns=label)
    y = df[label]

    return X, y
```


```python
test = preprocessed_data.copy()
x, y = split_sets(test, 'outcome')
```


```python
#type(x) == pd.DataFrame
type(y) == pd.Series
```




    True




```python
def temporal_data_split(data: pd.DataFrame, validation_days: int = 10, test_days: int = 10, label: str = 'outcome'):
    '''
    Perform the temporal data splitting into train, validation, and test set.

    Parameters:
    - data : pd.DataFrame
    - validation_days (int): Number of days for the validation set (default: 10).
    - test_days (int): Number of days for the test set (default: 10).
    - label (str): Name of the outcome variable (default: 'outcome').

    Returns:
    - pd.DataFrame: Training set features.
    - pd.Series: Training set outcome.
    - pd.DataFrame: Validation set features.
    - pd.Series: Validation set outcome.
    - pd.DataFrame: Test set features.
    - pd.Series: Test set outcome.
    '''
    try:
        # We confirm that the format of order_date is a datetime and group orders by same date
        try:
            data['order_date'] = pd.to_datetime(data['order_date']).dt.date
        except KeyError as ke:
            logging.error(f"Key Error: {str(ke)}")
            return 
        daily_orders = data.groupby('order_date').order_id.nunique()

        start_date = daily_orders.index.min()
        end_date = daily_orders.index.max()
        logging.info(f"Date from: {start_date}")
        logging.info(f"Date to: {end_date}")
        
        # Based on the number of days, we get the train days
        num_days = len(daily_orders)
        train_days = num_days - test_days - validation_days

        # Train
        train_start = daily_orders.index.min()
        train_end = train_start + timedelta(days = train_days)
        
        # Validation (no need to define test)
        validation_end = train_end + timedelta(days = validation_days)

        # Defined the cols finally used for model
        cols = numerical + binary + predicted

        train = data[data.order_date <= train_end][cols]
        val = data[(data.order_date > train_end) & (data.order_date <= validation_end)][cols]
        test = data[data.order_date > validation_end][cols]

        logging.info(f"Train set ratio: {len(train) / len(data):.2%}")
        logging.info(f"Validation set ratio: {len(val) / len(data):.2%}")
        logging.info(f"Test set ratio: {len(test) / len(data):.2%}")
        
        X_train, y_train = split_sets(train, label)
        X_val, y_val = split_sets(val, label)
        X_test, y_test = split_sets(test, label)

        return X_train, y_train, X_val, y_val, X_test, y_test
    
    # not the best practice to catch all exceptio
    except KeyError as ke:
        logging.error(f"Key Error: {str(ke)}")
        return
```


```python
X_train, y_train, X_val, y_val, X_test, y_test =  temporal_data_split(test)
```

    INFO - Date from: 2020-10-05
    INFO - Date to: 2021-03-03
    INFO - Train set ratio: 73.44%
    INFO - Validation set ratio: 13.32%
    INFO - Test set ratio: 13.24%



```python
def generate_evaluation_curves(model_name: str, C: float, y_pred, y_test, save_curves_path: str = None):
    '''
    Generate ROC and Precision-Recall curves for a binary classification model and save them in a single figure.

    Parameters:
    - model_name (str): Name of the model for labeling the curves.
    - y_pred (array-like): Predicted probabilities or scores.
    - y_test (array-like): True labels.
    - save_dir (str, optional): Directory to save the generated figure. If None, the figure will not be saved.

    Returns:
    - None
    '''

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f}) - {model_name} (C = {C:.2e})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.2f}) - {model_name} (C = {C:.2e})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()

    if save_curves_path:
        # Define the filename with a timestamp
        figure_filename = f'Evaluation_Curves_{model_name}_C={C}_{timestamp}.png'
        figure_path = os.path.join(save_curves_path, figure_filename)

        plt.savefig(figure_path)

    plt.show()
```


```python
save_curves_path = 'src/module_3/figures'
save_model_path = 'src/module_3/models'
```


```python
%cd ../..
!pwd
```

    /home/dan1dr/zrive-ds



```python
def train_evaluate_model(X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                        save_model_path: str):
    ''' 
    Train, evaluate, and save a logistic regression model with hyperparameter tuning for regularisation.
    Additionally, plot the curves for ROC and PR.

    Parameters:
    - X_train (pd.DataFrame): Training set features.
    - y_train (pd.Series): Training set outcome.
    - X_val (pd.DataFrame): Validation set features.
    - y_val (pd.Series): Validation set outcome.
    - save_model_path (str): Path to save the trained model.

    Returns:
    - None
    '''
    # Define parameter grid for C values
    # We know in advance that strong params works better
    param_grid = {
        'lr__C': np.logspace(-8, -2, num=9)
    }

    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(penalty='l2', solver='liblinear'))

    ])
    try:
        # Search for the best C hyperparameter
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best model and best parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Return probabilities for plotting the curves
        y_pred = best_model.predict_proba(X_val)[:, 1]

        # Print and save the evaluation curves (ROC and PR)
        generate_evaluation_curves(f'LogisticRegression', best_params['lr__C'], y_pred, y_val, save_curves_path)

        # Save the best model
        model_filename = f'LogisticRegression_{best_params["lr__C"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}.joblib'
        model_path = os.path.join(save_model_path, model_filename)
        joblib.dump(best_model, model_path)

        logging.info(f'Model {model_filename} saved  successfully!')

    except ValueError as e:
        logging.error(f'ValueError: {str(e)}')

    except pickle.PickleError as e:
        logging.error(f'Error during model save: {str(e)}')
```


```python
train_evaluate_model(X_train, y_train, X_val, y_val, save_model_path)
```


    
![png](push_exploration_files/push_exploration_163_0.png)
    


    INFO - Model LogisticRegression_1e-05_20231021124050.joblib saved  successfully!


Now, with the schema in mind for the MVP code in production, we will build:

1. def read + validate data
2. def preprocessing data
3. def temporal_split -> incorporate also the split_sets
4. def get_roc_pr_curves
5. def train and evaluate model, also save the model for later inference. Find some manner to save the best models choosed and history version

