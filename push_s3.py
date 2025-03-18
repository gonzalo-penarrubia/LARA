import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config


# Constants
BUCKET_NAME = 'Lara-bucket'

# S3 Client
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:4566',
    region_name='eu-west-1',
    config=Config(signature_version=UNSIGNED)
)


def upload(local_file_path: str, bucket_file_path: str, retry: int=0) -> None:
    '''
    Upload a file to s3. Create Bucket if not exists.

    Args:
        local_file_path (str): The local path of the file to upload.
        bucket_file_path (str): The path where the file has to be uploaded.
        retry (int): The number of retries the function has to end successfully.
    '''
    if retry < 3:
        try:
            for dirpath, dirnames, filenames in os.walk(local_file_path):
                s3.upload_file(dirpath, BUCKET_NAME, f'{bucket_file_path}/{dirpath}')
                print(f'Uploaded {filenames} successfully')

        except boto3.exceptions.S3UploadFailedError as e:
            if 'NoSuchBucket' in str(e):
                print('Creating bucket...')
                s3.create_bucket(Bucket=BUCKET_NAME)
                upload(retry=retry+1)

        except Exception as e:
            print(e)
            upload(retry=retry+1)
    else:
        print('Unexpected Error. The file couldn\'t be uploaded.')


if __name__ == '__main__':

    # (1) Push Model to S3
    print("Subiendo modelo a S3...")
    upload(
        local_file_path='./mlartifacts',
        bucket_file_path='models/'
    )

    print("Modelo subido con éxito.")