import boto3
import os

bedrock = boto3.client(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    service_name='bedrock', 
    region_name='us-east-1'
)

response = bedrock.list_foundation_models()
model_summaries = response['modelSummaries']
for model_summary in model_summaries:
    for key, value in model_summary.items():
        print(f'{key}: {value}')
    print('')
