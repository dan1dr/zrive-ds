# Part 1


```python
# Import libraries

import boto3
import urllib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
```


```python
#Disable warning for cleaning the terminal
urllib3.disable_warnings()

# We will use Amazon S3. We create a high-level resource object
# for interacting with AWS

#Adding verify = False as same issue in previous modules with corporate proxy
s3 = boto3.resource('s3',
                aws_access_key_id='AKIAXN64CPXK2VXBMPW3',
                aws_secret_access_key='',
                verify = False)

bucket_name = 'zrive-ds-data'
prefix = 'groceries/sampled-datasets/'

# Create a bucket object
bucket = s3.Bucket(bucket_name)

# Empty dict in case we need to iterate over
dfs = {}

#Iterate through the objects inside
for obj in bucket.objects.filter(Prefix = prefix):
    key = obj.key

    # We will keep a list of dfs in case we need to iterate over
    if key.endswith('.parquet'):
        print(f"-- Reading Parquet file: {key}")
        
        try:
            # Get the S3 object
            s3_object = s3.Object(bucket_name, key)

            # Get the parquet file as bytes
            response = s3_object.get()
            parquet_bytes = response['Body'].read()

            # Create a BytesIO object for seeking
            parquet_io = BytesIO(parquet_bytes)

            # We retrieve the actual filename
            df_name = key.split('/')[-1].split('.')[0]

            # Save as individual files and also inside a dict
            df = globals()[f'df_{df_name}'] = pd.read_parquet(parquet_io)
            dfs[df_name] = df

            print(f"The number of cols and rows is: {df.shape}")
            print(df.head())
            print(f"-- The df named df_{df_name} has been saved\n")

        except IOError as io_err:
            print(f"IOError reading {key}: {io_err}")
        except pd.errors.ParserError as parser_err:
            print(f"ParserError reading {key}: {parser_err}")
        except TypeError as type_err:
            if "a bytes-like object is required, not 'str'" in str(type_err):
                    print(f"TypeError: The Parquet file {key} is not in bytes format.")
            else:
                    print(f"TypeError reading {key}: {type_err}")
```

    -- Reading Parquet file: groceries/sampled-datasets/abandoned_carts.parquet
    The number of cols and rows is: (5457, 4)
                    id                                            user_id  \
    0   12858560217220  5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...   
    13  20352449839236  9d6187545c005d39e44d0456d87790db18611d7c7379bd...   
    45  20478401413252  e83fb0273d70c37a2968fee107113698fd4f389c442c0b...   
    50  20481783103620  10c42e10e530284b7c7c50f3a23a98726d5747b8128084...   
    52  20485321687172  d9989439524b3f6fc4f41686d043f315fb408b954d6153...   
    
                created_at                                         variant_id  
    0  2020-05-20 13:53:24  [33826459287684, 33826457616516, 3366719212762...  
    13 2021-06-27 05:24:13  [34415988179076, 34037940158596, 3450282236326...  
    45 2021-07-18 08:23:49  [34543001337988, 34037939372164, 3411360609088...  
    50 2021-07-18 21:29:36  [33667268116612, 34037940224132, 3443605520397...  
    52 2021-07-19 12:17:05   [33667268083844, 34284950454404, 33973246886020]  
    -- The df named df_abandoned_carts has been saved
    
    -- Reading Parquet file: groceries/sampled-datasets/inventory.parquet
    The number of cols and rows is: (1733, 6)
           variant_id  price  compare_at_price          vendor  \
    0  39587297165444      3                 3           heinz   
    1  34370361229444      5                 6   whogivesacrap   
    2  34284951863428      4                 4          plenty   
    3  33667283583108      2                 2  thecheekypanda   
    4  33803537973380      2                 2         colgate   
    
                          product_type                                        tags  
    0             condiments-dressings                       [table-sauces, vegan]  
    1  toilet-roll-kitchen-roll-tissue                 [b-corp, eco, toilet-rolls]  
    2  toilet-roll-kitchen-roll-tissue                              [kitchen-roll]  
    3  toilet-roll-kitchen-roll-tissue  [b-corp, cruelty-free, eco, tissue, vegan]  
    4                           dental                        [dental-accessories]  
    -- The df named df_inventory has been saved
    
    -- Reading Parquet file: groceries/sampled-datasets/orders.parquet
    The number of cols and rows is: (8773, 6)
                   id                                            user_id  \
    10  2204073066628  62e271062eb827e411bd73941178d29b022f5f2de9d37f...   
    20  2204707520644  bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...   
    21  2204838822020  329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...   
    34  2208967852164  f6451fce7b1c58d0effbe37fcb4e67b718193562766470...   
    49  2215889436804  68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...   
    
                created_at order_date  user_order_seq  \
    10 2020-04-30 14:32:19 2020-04-30               1   
    20 2020-04-30 17:39:00 2020-04-30               1   
    21 2020-04-30 18:12:30 2020-04-30               1   
    34 2020-05-01 19:44:11 2020-05-01               1   
    49 2020-05-03 21:56:14 2020-05-03               1   
    
                                            ordered_items  
    10  [33618849693828, 33618860179588, 3361887404045...  
    20  [33618835243140, 33618835964036, 3361886244058...  
    21  [33618891145348, 33618893570180, 3361889766618...  
    34  [33618830196868, 33618846580868, 3361891234624...  
    49  [33667166699652, 33667166699652, 3366717122163...  
    -- The df named df_orders has been saved
    
    -- Reading Parquet file: groceries/sampled-datasets/regulars.parquet
    The number of cols and rows is: (18105, 3)
                                                  user_id      variant_id  \
    3   68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...  33618848088196   
    11  aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...  33667178659972   
    18  68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...  33619009208452   
    46  aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...  33667305373828   
    47  4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...  33667247341700   
    
                created_at  
    3  2020-04-30 15:07:03  
    11 2020-05-05 23:34:35  
    18 2020-04-30 15:07:03  
    46 2020-05-05 23:34:35  
    47 2020-05-06 14:42:11  
    -- The df named df_regulars has been saved
    
    -- Reading Parquet file: groceries/sampled-datasets/users.parquet
    The number of cols and rows is: (4983, 10)
                                                    user_id user_segment  \
    2160  0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...       Top Up   
    1123  15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...       Top Up   
    1958  33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...       Top Up   
    675   57ca7591dc79825df0cecc4836a58e6062454555c86c35...       Top Up   
    4694  085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...       Top Up   
    
         user_nuts1     first_ordered_at customer_cohort_month  count_people  \
    2160        UKH  2021-05-08 13:33:49   2021-05-01 00:00:00           NaN   
    1123        UKJ  2021-11-17 16:30:20   2021-11-01 00:00:00           NaN   
    1958        UKD  2022-03-09 23:12:25   2022-03-01 00:00:00           NaN   
    675         UKI  2021-04-23 16:29:02   2021-04-01 00:00:00           NaN   
    4694        UKJ  2021-11-02 13:50:06   2021-11-01 00:00:00           NaN   
    
          count_adults  count_children  count_babies  count_pets  
    2160           NaN             NaN           NaN         NaN  
    1123           NaN             NaN           NaN         NaN  
    1958           NaN             NaN           NaN         NaN  
    675            NaN             NaN           NaN         NaN  
    4694           NaN             NaN           NaN         NaN  
    -- The df named df_users has been saved
    



```python
# We have saved the 5 dfs inside a dict called dfs
# We have saved each df with its name df_{parquet_file}
```

# EDA (in progress)

We have been presented with different datasets from a groceries e-commerce platform selling directly to consumers. In that sense, we have 5 different tables with different info in each of them.

- df_orders: An order history of customers. Each row is an order and the
item_ids for the order are stored as a list in the item_ids column
- df_regulars:  Users are allowed to specify items that they wish to buy
regularly. This data gives the items each user has asked to get regularly, along
with when they input that information.
- df_abandoned cart: If a user has added items to their basket but not
bought them, we capture that information. Items that were abandoned are stored
as a list in item_ids.
- df_inventory: Some information about each item_id
- df_users: Information about users.

1. Priorly to working with data, we will try to inspect each of the dataframes in order to understand the information. With the description of each table provided, we would try to figure out how the data looks like - then we could start thinking of some hypothesis and proceed with its verification.

2. We may display them, inspect columns and dtypes, verify if everythings makes sense about the metadata. After that, we could try to look for possible NANs or Null values, doublechecking data quality is OK. If we find any issue or problem during the process, with the prior knowledge obtained, we should be able to make a decision and justify the solution applied.

3. Once data has been fixed, we may join or transform them in order to work with that in future steps.

3. After all, we will formulate the corresponding hypothesis tested about the data plus some commentaries and additional info we have found.

### Orders

An orders history of customers. Each row is an order and the
item_ids for the order are stored as a list in the item_ids column



```python
print(df_orders.shape)
df_orders.head()
```

    (8773, 6)





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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>



### Regulars

Users are allowed to specify items that they wish to buy
regularly. This data gives the items each user has asked to get regularly, along
with when they input that information.


```python
print(df_regulars.shape)
df_regulars.head()
```

    (18105, 3)





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
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
    </tr>
  </tbody>
</table>
</div>



### Abandoned cart

If a user has added items to their basket but not
bought them, we capture that information. Items that were abandoned are stored
as a list in item_ids.



```python
print(df_abandoned_carts.shape)
df_abandoned_carts.head()
```

    (5457, 4)





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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>



### Inventory parquet

Some info about each item id


```python
print(df_inventory.shape)
df_inventory.head()
```

    (1733, 6)





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
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>



### Users

Information about users


```python
print(df_users.shape)
df_users.head()
```

    (4983, 10)





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
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c35...</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Assess general info about NANs


```python
def assess_NA(data):
    """
    Returns a pandas dataframe denoting the total number of NA
    values and the percentage of NA values in each column.
    The column names are noted on the index.
    """
    # pandas series denoting features and the sum of their null values
    null_sum = data.isnull().sum()# instantiate columns for missing data
    total = null_sum.sort_values(ascending=False)
    percent = ( ((null_sum / len(data.index))*100).round(2) ).sort_values(ascending=False)
    
    # concatenate along the columns to create the complete dataframe
    df_NA = pd.concat([total, percent], axis=1, sort=True, keys=['Number of NA', 'Percent NA'])

    return df_NA
```


```python
for df_name, df in dfs.items():
    """
    We will print general info about each df + apply assess_NA
    """
    print(f"--------- FILENAME: df_{df_name}\n")

    print(df.info(), "\n")
    print(df.describe(), "\n")

    result = assess_NA(df)
    print(result.head(),"\n\n")
```

    --------- FILENAME: df_abandoned_carts
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 5457 entries, 0 to 70050
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   id          5457 non-null   int64         
     1   user_id     5457 non-null   object        
     2   created_at  5457 non-null   datetime64[ns]
     3   variant_id  5457 non-null   object        
    dtypes: datetime64[ns](1), int64(1), object(2)
    memory usage: 213.2+ KB
    None 
    
                     id                     created_at
    count  5.457000e+03                           5457
    mean   2.161881e+13  2021-12-20 11:07:10.198460672
    min    1.285856e+13            2020-05-20 13:53:24
    25%    2.133401e+13            2021-11-13 19:52:17
    50%    2.167062e+13            2021-12-27 13:14:57
    75%    2.192303e+13            2022-01-30 08:35:19
    max    2.223385e+13            2022-03-13 14:12:10
    std    4.028679e+11                            NaN 
    
                Number of NA  Percent NA
    created_at             0         0.0
    id                     0         0.0
    user_id                0         0.0
    variant_id             0         0.0 
    
    
    --------- FILENAME: df_inventory
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB
    None 
    
             variant_id        price  compare_at_price
    count  1.733000e+03  1733.000000       1733.000000
    mean   3.694880e+13     6.307351          7.028881
    std    2.725674e+12     7.107218          7.660542
    min    3.361529e+13     0.000000          0.000000
    25%    3.427657e+13     2.490000          2.850000
    50%    3.927260e+13     3.990000          4.490000
    75%    3.948318e+13     7.490000          8.210000
    max    4.016793e+13    59.990000         60.000000 
    
                      Number of NA  Percent NA
    compare_at_price             0         0.0
    price                        0         0.0
    product_type                 0         0.0
    tags                         0         0.0
    variant_id                   0         0.0 
    
    
    --------- FILENAME: df_orders
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[ns]
     3   order_date      8773 non-null   datetime64[ns]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[ns](2), int64(2), object(2)
    memory usage: 479.8+ KB
    None 
    
                     id                     created_at  \
    count  8.773000e+03                           8773   
    mean   3.684684e+12  2021-08-22 03:54:18.750028288   
    min    2.204073e+12            2020-04-30 14:32:19   
    25%    3.690255e+12            2021-04-25 11:50:37   
    50%    3.846692e+12            2021-10-11 11:29:44   
    75%    3.945086e+12            2022-01-03 18:14:23   
    max    4.026732e+12            2022-03-14 00:24:59   
    std    4.145437e+11                            NaN   
    
                              order_date  user_order_seq  
    count                           8773     8773.000000  
    mean   2021-08-21 12:47:21.262966016        2.445116  
    min              2020-04-30 00:00:00        1.000000  
    25%              2021-04-25 00:00:00        1.000000  
    50%              2021-10-11 00:00:00        1.000000  
    75%              2022-01-03 00:00:00        3.000000  
    max              2022-03-14 00:00:00       25.000000  
    std                              NaN        2.707693   
    
                   Number of NA  Percent NA
    created_at                0         0.0
    id                        0         0.0
    order_date                0         0.0
    ordered_items             0         0.0
    user_id                   0         0.0 
    
    
    --------- FILENAME: df_regulars
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[ns]
    dtypes: datetime64[ns](1), int64(1), object(1)
    memory usage: 565.8+ KB
    None 
    
             variant_id                     created_at
    count  1.810500e+04                          18105
    mean   3.511989e+13  2021-08-15 02:27:30.703728384
    min    3.361527e+13            2020-04-30 13:09:27
    25%    3.382643e+13            2021-03-21 10:41:42
    50%    3.422171e+13            2021-10-16 09:11:26
    75%    3.448855e+13            2022-01-14 22:35:14
    max    4.013362e+13            2022-03-14 07:49:24
    std    2.171237e+12                            NaN 
    
                Number of NA  Percent NA
    created_at             0         0.0
    user_id                0         0.0
    variant_id             0         0.0 
    
    
    --------- FILENAME: df_users
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB
    None 
    
           count_people  count_adults  count_children  count_babies  count_pets
    count    325.000000    325.000000      325.000000    325.000000  325.000000
    mean       2.787692      2.003077        0.707692      0.076923    0.636923
    std        1.365753      0.869577        1.026246      0.289086    0.995603
    min        0.000000      0.000000        0.000000      0.000000    0.000000
    25%        2.000000      2.000000        0.000000      0.000000    0.000000
    50%        3.000000      2.000000        0.000000      0.000000    0.000000
    75%        4.000000      2.000000        1.000000      0.000000    1.000000
    max        8.000000      7.000000        6.000000      2.000000    6.000000 
    
                    Number of NA  Percent NA
    count_adults            4658       93.48
    count_babies            4658       93.48
    count_children          4658       93.48
    count_people            4658       93.48
    count_pets              4658       93.48 
    
    


At a first glance, everything looks correct. There is a specific group of columns count_{} in df_users (from column index = 5 until the end) that will be worthy to take a look:


```python
for columns in df_users.columns[5:]:
        print(df_users[columns].value_counts())
```

    count_people
    2.0    97
    3.0    68
    4.0    67
    1.0    57
    5.0    24
    6.0     8
    0.0     2
    7.0     1
    8.0     1
    Name: count, dtype: int64
    count_adults
    2.0    201
    1.0     71
    3.0     32
    4.0     11
    5.0      5
    0.0      3
    7.0      1
    6.0      1
    Name: count, dtype: int64
    count_children
    0.0    195
    2.0     58
    1.0     55
    3.0     12
    4.0      3
    5.0      1
    6.0      1
    Name: count, dtype: int64
    count_babies
    0.0    302
    1.0     21
    2.0      2
    Name: count, dtype: int64
    count_pets
    0.0    193
    1.0     87
    2.0     28
    3.0      8
    4.0      6
    5.0      2
    6.0      1
    Name: count, dtype: int64


Seems attributes for each user, where probably the user has the possibility to add more information about its household in his personal account (e.g. how many they are 'count_people', if they have children 'count_children', etc). The reason why there are so many nulls may be because it is not mandatory info, and people don't often fill up that info on their profiles

In that sense, we will keep that info and I do not see the need to drop them yet as the length of data is very manageable

After taking a look at all tables, seems that user_id it is the primary key across tables which identifies a single user registered in the platform. The attribute 'id' is it an unique identifier for each operation performed, such as abandon basket or purchases.

With the current magnitude of data, I see no problems for joining the df_orders (which contains the orders) with the inventory data using variant_id. Now, since we have several variant_id inside df_orders, we would need to unroll them and have one per row, but keeping items bought together with same identificator id. With that, we would have info about all the orders and its corresponding info from the inventory with price and type of product purchased.


```python
df_orders_unrolled = df_orders.explode(column='ordered_items', ignore_index=True)
df_orders_unrolled.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618849693828</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618860179588</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618874040452</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618907005060</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618907005060</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"Orders: {len(df_orders)}")
print(f"Total items: {len(df_orders_unrolled)}")

print(f"Avg items per order: {len(df_orders_unrolled)/len(df_orders):.2f}")
```

    Orders: 8773
    Total items: 107958
    Avg items per order: 12.31



```python
orders_with_inventory = df_orders_unrolled.merge(df_inventory, how='inner',left_on='ordered_items', right_on='variant_id')
orders_with_inventory.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.1</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.1</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2293056798852</td>
      <td>5842151f9684b07cb153383330d564d6b805488a4e7b4d...</td>
      <td>2020-06-06 13:03:00</td>
      <td>2020-06-06</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.1</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2293435203716</td>
      <td>228c4d57135f5610782353dee8709144543ad15216286f...</td>
      <td>2020-06-06 23:02:28</td>
      <td>2020-06-06</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.1</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2306493612164</td>
      <td>1296e1e72f7f43ff28d7d285f880ad4d213fa8139233c7...</td>
      <td>2020-06-21 15:09:26</td>
      <td>2020-06-21</td>
      <td>3</td>
      <td>33667238658180</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.1</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"Number of unique items in inventory: {df_inventory['variant_id'].nunique()}")
print(f"Unique items in inventory: {len(df_inventory)}")
print(f"Items sold merged with inventory: {len(orders_with_inventory)}")

```

    Number of unique items in inventory: 1733
    Unique items in inventory: 1733
    Items sold merged with inventory: 92361


We see the value length of orders_with_inventory changes depending which type of merge we perform (left, inner, right, outer). This means that are product being sold that have no corresponde on inventory and products in inventory not being sold. Would be interesting to check those values and verify with Sales team what could be happening there:

- Products in inventory not being sold might be a problem (OOS, wrong ids in one of the sides, horrible product that no one wants, etc)
- Products being sold but not in inventory (this is more problematic because it is normal that old products are not on sale anymore, but everything being sold should be tracked and maintained in their internal DB, so will be good to check if we are assigning wrong ids when we sell them or maybe there is some issue in inventory ids)

We could make some quick numbers on how many items are present whether in left or right hand


```python
left = df_orders_unrolled.merge(df_inventory, how='left',left_on='ordered_items', right_on='variant_id')
inner = df_orders_unrolled.merge(df_inventory, how='inner',left_on='ordered_items', right_on='variant_id')
right = df_orders_unrolled.merge(df_inventory, how='right',left_on='ordered_items', right_on='variant_id')
outer = df_orders_unrolled.merge(df_inventory, how='outer',left_on='ordered_items', right_on='variant_id')

print("All sold",len(left))
print("Inner",len(inner))
print("All inventory",len(right))
print("All together",len(outer))


```

    All sold 107958
    Inner 92361
    All inventory 92617
    All together 108214


As a first step, what I'll do is perform an outer join as I did in the previous step, but enable indicator=True and filter for right_only. Those are the missing products in inventory which are being sold (we are assuming here that sales are contemporary, another reason of this mismatch might be old sales that we are reading in our df_orders, PIPOs, phase out products, etc). Anyway, we can also check that in the date of the order we're reading. First, we will do the mentioned one:


```python
# Need to use this setting for display for avoiding scientific notation
pd.options.display.float_format = '{:.0f}'.format

inspect_orders_with_inventory = df_orders_unrolled.merge(df_inventory, how='outer',left_on='ordered_items', right_on='variant_id', indicator=True)

#Now we drop duplicates for same variant_id products
inspect_dropped = inspect_orders_with_inventory.drop_duplicates(subset=['variant_id'], keep='first')
#print(inspect_dropped['_merge'].value_counts())
missing = inspect_dropped[(inspect_dropped['_merge'] == 'left_only') | (inspect_dropped['_merge'] == 'right_only')]
print(missing['_merge'].value_counts())
```

    _merge
    right_only    256
    left_only       1
    both            0
    Name: count, dtype: int64



```python
missing.head()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
      <th>_merge</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618849693828</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>left_only</td>
    </tr>
    <tr>
      <th>107958</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39788282282116</td>
      <td>8</td>
      <td>8</td>
      <td>betteryou</td>
      <td>superfoods-supplements</td>
      <td>[cruelty-free, gluten-free, vitamins-supplements]</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>107959</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39349425242244</td>
      <td>6</td>
      <td>7</td>
      <td>nivea</td>
      <td>suncare</td>
      <td>[suncream]</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>107960</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40155145076868</td>
      <td>2</td>
      <td>2</td>
      <td>fairy</td>
      <td>dishwashing</td>
      <td>[washing-up-liquid]</td>
      <td>right_only</td>
    </tr>
    <tr>
      <th>107961</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
      <td>NaT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40070658490500</td>
      <td>0</td>
      <td>0</td>
      <td>jordans</td>
      <td>cereal</td>
      <td>[cereal-bars]</td>
      <td>right_only</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We do a quick manual check. This should not exist in orders dataframe
if not (df_orders_unrolled['ordered_items'] == '39788282282116').any():
    print("Not exist indeed")
```

    Not exist indeed


This means there are 256 unique ids in the inventory not being sold, and 1 sale with no correspondence in inventory. I would send this previous dataframe to Sales team in order to confirm it, but in this case we would observe that those might be de-listed EANs (past items in the inventory no longer available)

Having said that, until future confirmation from Sales team or extra-time we may dedicate, we would proceed with a left join for working with the data


```python
orders_with_inventory = df_orders_unrolled.merge(df_inventory, how='left',left_on='ordered_items', right_on='variant_id')
orders_with_inventory.tail()
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
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107953</th>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>34537169453188</td>
      <td>34537169453188</td>
      <td>1</td>
      <td>1</td>
      <td>whiskas</td>
      <td>cat-food</td>
      <td>[cat-treats]</td>
    </tr>
    <tr>
      <th>107954</th>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39368665432196</td>
      <td>39368665432196</td>
      <td>5</td>
      <td>5</td>
      <td>whiskas</td>
      <td>cat-food</td>
      <td>[wet-cat-food]</td>
    </tr>
    <tr>
      <th>107955</th>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39403033428100</td>
      <td>39403033428100</td>
      <td>3</td>
      <td>3</td>
      <td>napolina</td>
      <td>tins-packaged-foods</td>
      <td>[beans-pulses-lentils, discontinue, gluten-fre...</td>
    </tr>
    <tr>
      <th>107956</th>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39459281272964</td>
      <td>39459281272964</td>
      <td>4</td>
      <td>9</td>
      <td>cif</td>
      <td>cleaning-products</td>
      <td>[bathroom-limescale-cleaner, refills]</td>
    </tr>
    <tr>
      <th>107957</th>
      <td>4026732413060</td>
      <td>faaad4bc113fe7e0a888a0d3535877d3c692f591882c25...</td>
      <td>2022-03-14 00:24:59</td>
      <td>2022-03-14</td>
      <td>3</td>
      <td>39830155034756</td>
      <td>39830155034756</td>
      <td>4</td>
      <td>9</td>
      <td>cif</td>
      <td>cleaning-products</td>
      <td>[kitchen-oven-cleaner, refills]</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have the orders with products info, we can add additional info for each user (contained in df_users), about its regular wish list product (df_regulars), and about other items they dropped from their basket (df_abandoned)

What we would do for continuing inspecting orders is plot the data to know how many times a user ordered something (only with orders)


```python
# df_orders['user_order_seq'].value_counts().plot(kind='bar')
sns.countplot(data=df_orders_unrolled, x='user_order_seq', color='red')
```




    <Axes: xlabel='user_order_seq', ylabel='count'>




    
![png](module_2_files/module_2_37_1.png)
    


Now we continue inspecting orders_with_inventory


```python
print(f"% missing products in inventory: {100*orders_with_inventory.price.isna().sum() / len(orders_with_inventory):.2f}%")
```

    % missing products in inventory: 14.45%


We will estimate what was the most popular vendor and product-types


```python
orders_inventory_cleaned = orders_with_inventory.drop_duplicates(subset=['id', 'vendor']).groupby(['vendor'])[['id']].nunique()
orders_inventory_cleaned.sort_values(ascending=False, by='id', inplace=True)
orders_inventory_cleaned.head()
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
      <th>id</th>
    </tr>
    <tr>
      <th>vendor</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>oatly</th>
      <td>2023</td>
    </tr>
    <tr>
      <th>ecover</th>
      <td>1784</td>
    </tr>
    <tr>
      <th>fairy</th>
      <td>1738</td>
    </tr>
    <tr>
      <th>method</th>
      <td>1544</td>
    </tr>
    <tr>
      <th>andrex</th>
      <td>1397</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(data=orders_inventory_cleaned.head(10), x='vendor', y='id', color='red')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
     [Text(0, 0, 'oatly'),
      Text(1, 0, 'ecover'),
      Text(2, 0, 'fairy'),
      Text(3, 0, 'method'),
      Text(4, 0, 'andrex'),
      Text(5, 0, 'biona'),
      Text(6, 0, 'heinz'),
      Text(7, 0, 'mutti'),
      Text(8, 0, 'napolina'),
      Text(9, 0, 'persil')])




    
![png](module_2_files/module_2_42_1.png)
    



```python
orders_inventory_cleaned = orders_with_inventory.drop_duplicates(subset=['id', 'product_type']).groupby(['product_type'])[['id']].nunique()
orders_inventory_cleaned.sort_values(ascending=False, by='id', inplace=True)
orders_inventory_cleaned.head()
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
      <th>id</th>
    </tr>
    <tr>
      <th>product_type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cleaning-products</th>
      <td>3500</td>
    </tr>
    <tr>
      <th>tins-packaged-foods</th>
      <td>3281</td>
    </tr>
    <tr>
      <th>toilet-roll-kitchen-roll-tissue</th>
      <td>3131</td>
    </tr>
    <tr>
      <th>long-life-milk-substitutes</th>
      <td>2657</td>
    </tr>
    <tr>
      <th>dishwashing</th>
      <td>2632</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(data=orders_inventory_cleaned.head(10), x='product_type', y='id', color='red')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

```




    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
     [Text(0, 0, 'cleaning-products'),
      Text(1, 0, 'tins-packaged-foods'),
      Text(2, 0, 'toilet-roll-kitchen-roll-tissue'),
      Text(3, 0, 'long-life-milk-substitutes'),
      Text(4, 0, 'dishwashing'),
      Text(5, 0, 'snacks-confectionery'),
      Text(6, 0, 'cooking-ingredients'),
      Text(7, 0, 'soft-drinks-mixers'),
      Text(8, 0, 'condiments-dressings'),
      Text(9, 0, 'cereal')])




    
![png](module_2_files/module_2_44_1.png)
    



```python
# Replicated in matplotlib
orders_inventory_cleaned.head(10).plot(kind='bar', color='red')
```




    <Axes: xlabel='product_type'>




    
![png](module_2_files/module_2_45_1.png)
    


As typically in the FMCG sector, we see a dominace of Home Care products over the rest of categories. We also appreciate the overpresence of some eco/healthy/bio products, which could give us some clue about the type of groceries we're analyzing and the avg customer we could expect (maybe typical middle-high class UK which pays a premium for those products). We could gather those hypothesis and test it against more customer info, but we will leave here for continuing with some more patterns from orders.

Now let's try to see the evolution for items and orders over time


```python
orders_count = orders_with_inventory.groupby('order_date')['id'].count()
plt.figure(figsize=(10, 6))
plt.plot(orders_count.index, orders_count.values, marker='.', color='orange')
```




    [<matplotlib.lines.Line2D at 0x7f8c6b833b10>]




    
![png](module_2_files/module_2_48_1.png)
    



```python
# Too many points on a daily basis. Let's group them by weeks
orders_with_inventory['order_date'] = pd.to_datetime(orders_with_inventory['order_date'])
orders_count = orders_with_inventory.groupby(orders_with_inventory['order_date'].dt.to_period("W"))['id'].count()
plt.plot(orders_count.index.to_timestamp(), orders_count.values, marker='.', color='orange')
```




    [<matplotlib.lines.Line2D at 0x7f8c6d912a10>]




    
![png](module_2_files/module_2_49_1.png)
    



```python
#Playing here with another version of plotting the data with pandas
orders_with_inventory['order_date'] = pd.to_datetime(orders_with_inventory['order_date'])

# Extract week and year from 'order_date'
orders_with_inventory['week'] = orders_with_inventory['order_date'].dt.isocalendar().week
orders_with_inventory['year'] = orders_with_inventory['order_date'].dt.isocalendar().year

# Group by 'year' and 'week', and count unique 'id' values
historical_evolution = orders_with_inventory.groupby(['year', 'week'])['id'].nunique()
historical_evolution.plot(kind='line', marker='.', color='orange')
```




    <Axes: xlabel='year,week'>




    
![png](module_2_files/module_2_50_1.png)
    



```python
#And now a lit bit nicer
plt.figure(figsize=(10, 6))
historical_evolution.plot(kind='line', marker='.', color='orange')
plt.xlabel('Year-Week')
plt.ylabel('Orders')
plt.title('Orders per week')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
```


    
![png](module_2_files/module_2_51_0.png)
    



```python
# Convert 'order_date' column to datetime
orders_with_inventory['order_date'] = pd.to_datetime(orders_with_inventory['order_date'])

# Extract week and year from 'order_date'
orders_with_inventory['week'] = orders_with_inventory['order_date'].dt.isocalendar().week
orders_with_inventory['year'] = orders_with_inventory['order_date'].dt.isocalendar().year

# Group by 'year' and 'week', and count 'ordered_items'
historical_evolution = orders_with_inventory.groupby(['year', 'week'])['ordered_items'].count()

# Create a plot
plt.figure(figsize=(10, 6))
historical_evolution.plot(kind='line', marker='.', color='orange')
plt.xlabel('Year-Week')
plt.ylabel('Total Ordered Items')
plt.title('Total Ordered Items per week')
plt.xticks(rotation=45)
plt.grid(True)
```


    
![png](module_2_files/module_2_52_0.png)
    


Obviously, the plot is almost the same but scaled by approx avg items per orders (~x12, as we saw) :D.

Maybe, would be useful to plot the evolution of the items per basket over time to see if basket has increased its size along time (also we could plot the avg basket value )


```python
orders_with_inventory['order_date'] = pd.to_datetime(orders_with_inventory['order_date'])

weekly_avg_ordered_items  = orders_with_inventory.groupby(['order_date', 'id'])['ordered_items'].count().reset_index()

weekly_avg_ordered_items = weekly_avg_ordered_items.groupby(weekly_avg_ordered_items['order_date'].dt.to_period("W"))['ordered_items'].mean()
plt.figure(figsize=(8, 4))
weekly_avg_ordered_items.plot(kind='line', marker='.', color='orange')
plt.xlabel('Year-Week')
plt.ylabel('Time')
plt.title('Avg items per basket over time')


```




    Text(0.5, 1.0, 'Avg items per basket over time')




    
![png](module_2_files/module_2_54_1.png)
    


We see a decrease in number of items per basket. We could argue that people have decided to buy less items, possibly motivated by change of inventory from the store. The store might have reduced the total assortment simplifying the total items on sale. This is a typical pattern in some retailers, where they perform assortment optimization after researching the optimal numbers of EANs in their portfolio. Let's see if that has been materialized in the total value $ over time.


```python
value_time = orders_with_inventory.groupby(orders_with_inventory['order_date'].dt.to_period("W"))[['price']].sum()

plt.figure(figsize=(10, 6))
value_time.plot(kind='line', marker=".", color='orange')
plt.xlabel('Time')
plt.ylabel('£')
plt.title('Volume (£) over time')

#Fig size does not adapt here idk why

```




    Text(0.5, 1.0, 'Volume (£) over time')




    <Figure size 1000x600 with 0 Axes>



    
![png](module_2_files/module_2_56_2.png)
    


Volume value indeed has been increasing since beginning. We see a peak Jan-Feb'22 that could be interpreted (event, promo, discount, etc), and then a sudden crease in the last weeks until today - most probably bc data was not yet collected for that last one. This graphic is, again, very similar to orders and avg items over time as it is highly correlated. Let's take a look at the avg basket value over time finally:


```python
avg_value = orders_with_inventory.dropna(subset='price').groupby(['id', 'order_date'])['price'].sum().reset_index()
avg_value = avg_value.groupby(avg_value['order_date'].dt.to_period("W"))['price'].mean()

plt.figure(figsize=(10, 6))
avg_value.plot(kind='line', marker=".", color='orange')
plt.xlabel('Time')
plt.ylabel('£')
plt.title('Avg basket value over time')

```




    Text(0.5, 1.0, 'Avg basket value over time')




    
![png](module_2_files/module_2_58_1.png)
    



```python
avg_value = orders_with_inventory.dropna(subset='price').groupby(['id', 'order_date'])['price'].sum().reset_index()
avg_value[avg_value['order_date'] > '2020-01-01']['price'].mean()
```




    56.13289594315838



So the value has been steadily increasing until today, where it has been more or less stabilized around £55

## Users


```python
df_users.head()
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
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c35...</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(f"Number of unique customers: {df_users['user_id'].nunique()}")
```

    Number of unique customers: 4983



```python
df_users.describe()
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
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>325</td>
      <td>325</td>
      <td>325</td>
      <td>325</td>
      <td>325</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group and count by week (you can change the frequency as needed)
df_users['first_ordered_at'] = pd.to_datetime(df_users['first_ordered_at'])
customer_acq = df_users.resample('W', on='first_ordered_at')['user_id'].count()

plt.figure(figsize=(10, 6))
customer_acq.plot(kind='line', marker='.', color='lime')
plt.xlabel('Time')
plt.ylabel('Customers')
plt.title('Customers registered over time')


# Calculate the cumulative sum of customers acquired
cumulative_customer_acq = customer_acq.cumsum()
plt.figure(figsize=(10, 6))
cumulative_customer_acq.plot(kind='line', marker='.', color='green')
plt.xlabel('Time')
plt.ylabel('Customers')
plt.title('Cumulative customers over time')


```




    Text(0.5, 1.0, 'Cumulative customers over time')




    
![png](module_2_files/module_2_65_1.png)
    



    
![png](module_2_files/module_2_65_2.png)
    


We see the peak that we observed before for some special event at the beginning of 2022

## Regulars

We will merge it with inventory for having product info


```python
#Dropping nans as we would like to inspect for those available
regulars = df_regulars.merge(df_inventory, on='variant_id', how='left').dropna()
regulars.head()
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
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
      <td>3</td>
      <td>4</td>
      <td>method</td>
      <td>cleaning-products</td>
      <td>[cruelty-free, eco, vegan, window-glass-cleaner]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667182493828</td>
      <td>2020-05-06 14:42:11</td>
      <td>4</td>
      <td>5</td>
      <td>bulldog</td>
      <td>skincare</td>
      <td>[cruelty-free, eco, facial-skincare, vegan]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d883991facbc3b07b62da342d00c97d1e6cea8d2176695...</td>
      <td>33667198910596</td>
      <td>2020-07-06 10:12:08</td>
      <td>15</td>
      <td>17</td>
      <td>ecover</td>
      <td>dishwashing</td>
      <td>[cruelty-free, dishwasher-tablets, eco, vegan]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>66a195720d6988ff4d32155cc03631b84f68b34d3b0a1e...</td>
      <td>33826459320452</td>
      <td>2020-07-06 17:17:52</td>
      <td>5</td>
      <td>6</td>
      <td>treeoflife</td>
      <td>snacks-confectionery</td>
      <td>[christmas, nuts-dried-fruit-seeds, organic]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0b7e02fee4b9e215da3bdae70050f20c5ffd18264454a5...</td>
      <td>33667247276164</td>
      <td>2020-07-18 16:56:55</td>
      <td>2</td>
      <td>3</td>
      <td>method</td>
      <td>hand-soap-sanitisers</td>
      <td>[cruelty-free, eco, hand-soap, vegan]</td>
    </tr>
  </tbody>
</table>
</div>




```python
most_regulars_vendor = regulars.groupby('vendor')[['user_id']].count().sort_values(by='user_id',ascending=False)
most_regulars_vendor.head(10).plot(kind='bar', color='pink')


most_regulars_product_type = regulars.groupby('product_type')[['user_id']].count().sort_values(by='user_id',ascending=False)
most_regulars_product_type.head(10).plot(kind='bar', color='purple')
```




    <Axes: xlabel='product_type'>




    
![png](module_2_files/module_2_70_1.png)
    



    
![png](module_2_files/module_2_70_2.png)
    



```python
#Which can be also achieved by:
vendors = regulars['vendor'].value_counts().head(10)
vendors.plot(kind='bar', color='pink')
```




    <Axes: xlabel='vendor'>




    
![png](module_2_files/module_2_71_1.png)
    


Now we might superpone the most common regulars product type (i.e. expected avg distribution sales) vs actual_sales vs inventoryy


```python
orders_inventory_cleaned = orders_with_inventory.groupby(['product_type'])[['id']].count().sort_values(by='id', ascending=False)
orders_inventory_cleaned.head(10).plot(kind='bar', color='red')

inventory = df_inventory.groupby('product_type')[['variant_id']].count().sort_values(by='variant_id', ascending=False)
inventory.head(10).plot(kind='bar', color='cornflowerblue')

most_regulars_product_type = regulars.groupby('product_type')[['user_id']].count().sort_values(by='user_id',ascending=False)
most_regulars_product_type.head(10).plot(kind='bar', color='purple')
```




    <Axes: xlabel='product_type'>




    
![png](module_2_files/module_2_73_1.png)
    



    
![png](module_2_files/module_2_73_2.png)
    



    
![png](module_2_files/module_2_73_3.png)
    



```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

orders_inventory_cleaned.head(10).plot(kind='bar', color='red', ax=axes[0])
axes[0].set_title('Orders')
axes[0].set_xlabel('Product Type')
axes[0].set_ylabel('Count')

inventory.head(10).plot(kind='bar', color='cornflowerblue', ax=axes[1])
axes[1].set_title('Inventory')
axes[1].set_xlabel('Product Type')
axes[1].set_ylabel('Count')

most_regulars_product_type.head(10).plot(kind='bar', color='purple', ax=axes[2])
axes[2].set_title('Most Regulars')
axes[2].set_xlabel('Product Type')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.show()
```


    
![png](module_2_files/module_2_74_0.png)
    



```python
# Showing now the 3 together
combined_df = pd.concat([orders_inventory_cleaned,
                         inventory,
                         most_regulars_product_type], axis=1)
combined_df.columns = ['Top Orders', 'Top Inventory', 'Top Regulars']

#Rank them based on Top Orders ranking
combined_df = combined_df.sort_values(by='Top Orders', ascending=False).head(20)
combined_df.head(20)

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
      <th>Top Orders</th>
      <th>Top Inventory</th>
      <th>Top Regulars</th>
    </tr>
    <tr>
      <th>product_type</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cleaning-products</th>
      <td>9676</td>
      <td>160</td>
      <td>1877</td>
    </tr>
    <tr>
      <th>tins-packaged-foods</th>
      <td>8959</td>
      <td>125</td>
      <td>1402</td>
    </tr>
    <tr>
      <th>long-life-milk-substitutes</th>
      <td>6637</td>
      <td>24</td>
      <td>566</td>
    </tr>
    <tr>
      <th>toilet-roll-kitchen-roll-tissue</th>
      <td>6314</td>
      <td>18</td>
      <td>802</td>
    </tr>
    <tr>
      <th>dishwashing</th>
      <td>4362</td>
      <td>27</td>
      <td>834</td>
    </tr>
    <tr>
      <th>soft-drinks-mixers</th>
      <td>3808</td>
      <td>48</td>
      <td>482</td>
    </tr>
    <tr>
      <th>snacks-confectionery</th>
      <td>3787</td>
      <td>122</td>
      <td>660</td>
    </tr>
    <tr>
      <th>cooking-ingredients</th>
      <td>3150</td>
      <td>73</td>
      <td>784</td>
    </tr>
    <tr>
      <th>cereal</th>
      <td>3014</td>
      <td>51</td>
      <td>471</td>
    </tr>
    <tr>
      <th>condiments-dressings</th>
      <td>2821</td>
      <td>52</td>
      <td>521</td>
    </tr>
    <tr>
      <th>cooking-sauces</th>
      <td>2631</td>
      <td>43</td>
      <td>444</td>
    </tr>
    <tr>
      <th>pasta-rice-noodles</th>
      <td>2443</td>
      <td>66</td>
      <td>461</td>
    </tr>
    <tr>
      <th>fabric-softener-freshener</th>
      <td>2443</td>
      <td>17</td>
      <td>321</td>
    </tr>
    <tr>
      <th>dental</th>
      <td>2161</td>
      <td>42</td>
      <td>430</td>
    </tr>
    <tr>
      <th>biscuits-crackers</th>
      <td>2008</td>
      <td>49</td>
      <td>326</td>
    </tr>
    <tr>
      <th>hand-soap-sanitisers</th>
      <td>1981</td>
      <td>24</td>
      <td>388</td>
    </tr>
    <tr>
      <th>spreads</th>
      <td>1926</td>
      <td>19</td>
      <td>380</td>
    </tr>
    <tr>
      <th>home-baking</th>
      <td>1808</td>
      <td>25</td>
      <td>403</td>
    </tr>
    <tr>
      <th>haircare</th>
      <td>1552</td>
      <td>50</td>
      <td>299</td>
    </tr>
    <tr>
      <th>washing-liquid-gel</th>
      <td>1482</td>
      <td>18</td>
      <td>241</td>
    </tr>
  </tbody>
</table>
</div>




```python

# Plotting now the 3 together

# Calculate the total count across all product types
total_counts = combined_df.sum()

# Normalize the counts to percentages relative to the total count
normalized_df = (combined_df / total_counts) * 100

# Create an array of x positions for each group
x = np.arange(len(normalized_df))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 10))

for i, (column, color) in enumerate(zip(normalized_df.columns, ['red', 'cornflowerblue', 'purple'])):
    ax.bar(x + i * width, normalized_df[column], width=width, label=column, color=color)

ax.set_xticks(x + width)
ax.set_xticklabels(normalized_df.index, rotation=90)

ax.set_xlabel('Product Type')
ax.set_ylabel('Share')
ax.set_title('Share by Product Type')
ax.legend()

plt.tight_layout()
plt.show()

```


    
![png](module_2_files/module_2_76_0.png)
    


And here we see regulars it is a good proxy for sales, except for some milk substitutes (potentially a whim) and toilet rolls (the urgency surprass the expected). For the top2 product-types we are enough stocked, but for milk, toilet, and dishwashing we could point there is not enough inventory products as they are represented in orders. Moreover, we could optimize the assorment offered and improve our inventory distribution doing a lot of different techniques and hypothesis, taking a look at the overrepresented categories or items in inventory (like snacks, cooking or pasta) vs actual sales / regulars.

# Part 2


```python
# We will recreate the bucket in order to get used to getting resources from there

s3 = boto3.resource('s3',
                aws_access_key_id='AKIAXN64CPXK2VXBMPW3',
                aws_secret_access_key='',
                verify = False)

# Define again the bucket
bucket_name = 'zrive-ds-data'

# Create a bucket object
bucket = s3.Bucket(bucket_name)

file_key = 'groceries/box_builder_dataset/feature_frame.csv'
local_path = '/home/dan1dr/data/feature_frame.csv'

# Function to track % progress
def download_progress(chunk):
    global total_bytes, previous_percent_completed
    total_bytes += chunk
    percent_completed = int((total_bytes / total_size) * 100)
    if percent_completed - previous_percent_completed >=10:
        previous_percent_completed = percent_completed
        print(f"Downloaded: {total_bytes} bytes, {percent_completed}% complete")

# Initialize variables to track progress
total_bytes = 0
previous_percent_completed = 0

# Get the file size without downloading it
object_summary = s3.ObjectSummary(bucket_name, file_key)
total_size = object_summary.size

# Download the file with progress tracking
try:
    s3.Bucket(bucket_name).download_file(file_key, local_path, Callback=download_progress)
    print("Download complete!")
except Exception as e:
    print(f"Download failed: {str(e)}")


s3.Bucket(bucket_name).download_file(file_key, local_path)
```

    Downloaded: 76283904 bytes, 10% complete
    Downloaded: 152567808 bytes, 20% complete
    Downloaded: 228589568 bytes, 30% complete
    Downloaded: 304873472 bytes, 40% complete
    Downloaded: 380895232 bytes, 50% complete
    Downloaded: 457179136 bytes, 60% complete
    Downloaded: 533200896 bytes, 70% complete
    Downloaded: 609484800 bytes, 80% complete
    Downloaded: 685768704 bytes, 90% complete
    Downloaded: 761678715 bytes, 100% complete
    Download complete!



```python
#Now load the downloaded file as a df
df_groceries = pd.read_csv(local_path)
```


```python
df_groceries.head()
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



In this dataset, every row representes an (order, product) pair where outcome indicates whether the product was bought or not on that order and every other feature has been computed only looking at previous information about the user and the product (avoid information bias).


```python
len(df_groceries)
```




    2880549




```python
df_groceries.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB



```python
nans = assess_NA(df_groceries)
print(nans)
```

                                      Number of NA  Percent NA
    abandoned_before                             0           0
    active_snoozed                               0           0
    avg_days_to_buy_product_type                 0           0
    avg_days_to_buy_variant_id                   0           0
    count_adults                                 0           0
    count_babies                                 0           0
    count_children                               0           0
    count_pets                                   0           0
    created_at                                   0           0
    days_since_purchase_product_type             0           0
    days_since_purchase_variant_id               0           0
    discount_pct                                 0           0
    global_popularity                            0           0
    normalised_price                             0           0
    order_date                                   0           0
    order_id                                     0           0
    ordered_before                               0           0
    outcome                                      0           0
    people_ex_baby                               0           0
    product_type                                 0           0
    set_as_regular                               0           0
    std_days_to_buy_product_type                 0           0
    std_days_to_buy_variant_id                   0           0
    user_id                                      0           0
    user_order_seq                               0           0
    variant_id                                   0           0
    vendor                                       0           0



```python
df_groceries['outcome'].value_counts()
```




    outcome
    0.0    2847317
    1.0      33232
    Name: count, dtype: int64




```python
print(f"Bought: {len(df_groceries[df_groceries['outcome'] == 1])/len(df_groceries) *100:.2f}%")
```

    Bought: 1.15%


There are different columns in the data with several value types:
- Label: ['outcome']
- Information: ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
- Feature: All them except the above

Within all features, we can group them if:
- Numerical: ['user_order_seq', 'normalised_price', 'discount_pct', 'global_popularity', 'count_adults', 'count_children', 'count_babies', 'count_pets', 'people_ex_baby', 'days_since_purchase_variant_id', 'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id', 'days_since_purchase_product_type', 'avg_days_to_buy_product_type', 'std_days_to_buy_product_type']

- Categorical: (splitted by categorical and binary)
    - Categorical: ['product_type', 'vendor']
    - Binary: ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']


```python
# We define the columns by groups
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


```python
df_groceries[numerical].describe()
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
      <th>user_order_seq</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
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
      <th>count</th>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.289342e+00</td>
      <td>1.272808e-01</td>
      <td>1.862744e-01</td>
      <td>1.070302e-02</td>
      <td>2.017627e+00</td>
      <td>5.492182e-02</td>
      <td>3.538562e-03</td>
      <td>5.134091e-02</td>
      <td>2.072549e+00</td>
      <td>3.312961e+01</td>
      <td>3.523734e+01</td>
      <td>2.645304e+01</td>
      <td>3.143513e+01</td>
      <td>3.088810e+01</td>
      <td>2.594969e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.140176e+00</td>
      <td>1.268378e-01</td>
      <td>1.934480e-01</td>
      <td>1.663389e-02</td>
      <td>2.098915e-01</td>
      <td>3.276586e-01</td>
      <td>5.938048e-02</td>
      <td>3.013646e-01</td>
      <td>3.943659e-01</td>
      <td>3.707162e+00</td>
      <td>1.057766e+01</td>
      <td>7.168323e+00</td>
      <td>1.227511e+01</td>
      <td>4.330262e+00</td>
      <td>3.278860e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000e+00</td>
      <td>1.599349e-02</td>
      <td>-4.016064e-02</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.414214e+00</td>
      <td>0.000000e+00</td>
      <td>7.000000e+00</td>
      <td>2.828427e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000e+00</td>
      <td>5.394416e-02</td>
      <td>8.462238e-02</td>
      <td>1.628664e-03</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.000000e+01</td>
      <td>2.319372e+01</td>
      <td>3.000000e+01</td>
      <td>2.800000e+01</td>
      <td>2.427618e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000e+00</td>
      <td>8.105178e-02</td>
      <td>1.169176e-01</td>
      <td>6.284368e-03</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.400000e+01</td>
      <td>2.769305e+01</td>
      <td>3.000000e+01</td>
      <td>3.100000e+01</td>
      <td>2.608188e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000e+00</td>
      <td>1.352670e-01</td>
      <td>2.234637e-01</td>
      <td>1.418440e-02</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>4.000000e+01</td>
      <td>3.059484e+01</td>
      <td>3.000000e+01</td>
      <td>3.400000e+01</td>
      <td>2.796118e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.100000e+01</td>
      <td>1.000000e+00</td>
      <td>1.325301e+00</td>
      <td>4.254386e-01</td>
      <td>5.000000e+00</td>
      <td>3.000000e+00</td>
      <td>1.000000e+00</td>
      <td>6.000000e+00</td>
      <td>5.000000e+00</td>
      <td>1.480000e+02</td>
      <td>8.400000e+01</td>
      <td>5.868986e+01</td>
      <td>1.480000e+02</td>
      <td>3.950000e+01</td>
      <td>3.564191e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What we can do is for all numerical columns, try to see if there are any special correlations vs the predicted value (outcome):

df_corr = df_groceries[predicted + numerical]
corr_matrix = df_corr.corr()

plt.figure(figsize=(10, 8))

#Apply some visual styles. Show values, colors, scale, decimals showed, lines width.
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmax=0.5, fmt=".2f", linewidths=0.5)

plt.title('Correlation Heatmap')
plt.show()

```


    
![png](module_2_files/module_2_91_0.png)
    


Almost no correlation vs. outcome. Additionaly, some variables between themselves are highly correlated (like people_ex_baby vs. count adults-children-pets), so important to remember when applying some models when multicolinearity may represent a problem (difficulty in identifying important features, incorrect coefficients, reduced robustness). This problem it is usually addressed using techniques like feature selection, PCA, etc.

For categorical cols, we could take a look at them:


```python
df_groceries[categorical].describe()
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
      <th>product_type</th>
      <th>vendor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2880549</td>
      <td>2880549</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>62</td>
      <td>264</td>
    </tr>
    <tr>
      <th>top</th>
      <td>tinspackagedfoods</td>
      <td>biona</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>226474</td>
      <td>146828</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_groceries.groupby('product_type')['product_type'].count().sort_values(ascending=False)
```




    product_type
    tinspackagedfoods         226474
    condimentsdressings       129749
    ricepastapulses           128098
    haircare                  114978
    cookingingredientsoils    110686
                               ...  
    babyfood12months            6797
    householdsundries           6735
    petcare                     4075
    feedingweaning              2790
    premixedcocktails           2620
    Name: product_type, Length: 62, dtype: int64




```python
df_groceries.groupby('vendor')['vendor'].count().sort_values(ascending=False)

```




    vendor
    biona         146828
    ecover        113018
    method         79258
    organix        74632
    treeoflife     68920
                   ...  
    minky            599
    vitalbaby        594
    munchkin         422
    freee            336
    vicks            243
    Name: vendor, Length: 264, dtype: int64




```python
print(df_groceries['product_type'].nunique())
print(df_groceries['vendor'].nunique())
```

    62
    264



```python
for feature in df_groceries[categorical]:
    plt.figure(figsize=(14, 6))
    sns.countplot(data=df_groceries, x=feature, palette='Set3', hue=feature)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=90)
    plt.show()
```


    
![png](module_2_files/module_2_98_0.png)
    



    
![png](module_2_files/module_2_98_1.png)
    


Maybe use product_type as a categ variable because vendor has too many different values (we might group them in more general ones).

Let's take a look at binary cols


```python
for col in df_groceries[binary]:
    print(df_groceries[col].value_counts())
```

    ordered_before
    0.0    2819658
    1.0      60891
    Name: count, dtype: int64
    abandoned_before
    0.0    2878794
    1.0       1755
    Name: count, dtype: int64
    active_snoozed
    0.0    2873952
    1.0       6597
    Name: count, dtype: int64
    set_as_regular
    0.0    2870093
    1.0      10456
    Name: count, dtype: int64



```python
for col in df_groceries[binary]:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_groceries, x=col, palette='Set2', hue=col, legend=False)
    plt.title(f'Distribution of {col}')
    plt.show()
```


    
![png](module_2_files/module_2_102_0.png)
    



    
![png](module_2_files/module_2_102_1.png)
    



    
![png](module_2_files/module_2_102_2.png)
    



    
![png](module_2_files/module_2_102_3.png)
    



```python
plt.figure(figsize=(6, 4))
sns.countplot(data=df_groceries, x='outcome', palette='Set2')
plt.title('Distribution of Outcome')
plt.show()

```

    /tmp/ipykernel_465/2908886851.py:2: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(data=df_groceries, x='outcome', palette='Set2')



    
![png](module_2_files/module_2_103_1.png)
    


Very unbalanced features for binaries. We could think of addressing it during modelling because it can lead to biased model performance (i.e. apply resampling techiques, evaluation metrics, class weighting, etc)

Finally, we could plot the data of each value in order to inspect each outcome distribution. Depending on the technique we would like to apply, we might revisit the EDA in order to do some adjustments and so. It is an iterative process and a more extensive EDA might be applied with a bit of domain knowledge, iterating and refining the data if we had a pre-tought of the type of model we would like to apply later on.
