import boto3
import urllib3
import pandas as pd
from io import BytesIO


#Disable warning for cleaning the terminal
urllib3.disable_warnings()


# We will use Amazon S3. We create a high-level resource object
# for interacting with AWS


#Adding verify = False as same issue in previous modules with corporate proxy
s3 = boto3.resource('s3',
                aws_access_key_id='AKIAXN64CPXKVY56HGZZ',
                aws_secret_access_key='XYZ',
                verify = False)

bucket_name = 'zrive-ds-data'
prefix = 'groceries/sampled-datasets/'

# Create a bucket object
bucket = s3.Bucket(bucket_name)

#Iterate through the objects inside
for obj in bucket.objects.filter(Prefix = prefix):
    key = obj.key

    if key.endswith('.parquet'):
        print(f"Reading Parquet file: {key}")
        
        try:
            # Get the S3 object
            s3_object = s3.Object(bucket_name, key)

            # Get the parquet file as bytes
            response = s3_object.get()
            parquet_bytes = response['Body'].read()

            # Create a BytesIO object for seeking
            parquet_io = BytesIO(parquet_bytes)

            df = pd.read_parquet(parquet_io)
            print(df.head())

        except IOError as io_err:
            print(f"IOError reading {key}: {io_err}")
        except pd.errors.ParserError as parser_err:
            print(f"ParserError reading {key}: {parser_err}")
        except TypeError as type_err:
            if "a bytes-like object is required, not 'str'" in str(type_err):
                    print(f"TypeError: The Parquet file {key} is not in bytes format.")
            else:
                    print(f"TypeError reading {key}: {type_err}")


def main():
    pass

if __name__ == "__main__":
    main()

