import json
import boto3
import string
import os
import numpy as np
from encode import one_hot_encode
from encode import vectorize_sequences


def lambda_handler(event, context):
    
    s3 = boto3.client('s3')
    ses = boto3.client('ses')
    
    if event:
        
        file_obj = event['Records'][0]
        bucket = str(file_obj['s3']['bucket']['name'])
        
        filename = str(file_obj['s3']['object']['key'])
        #print('Filename: ', filename)
        
        file = s3.get_object(Bucket = bucket, Key = filename)
        content = file["Body"].read().decode('utf-8')
        
        cbody = content.split('Content-Type: text/plain; charset="UTF-8"')[1] 
        cbody = cbody.split('Content-Type: text/html; charset="UTF-8"')[0]
        cbody = cbody.rsplit("\n",3)[0]
        cbody = cbody.replace("\n"," ")
        
        subject = content.split('Subject: ')[1]
        subject = subject.split('\n')[0]

        edate = content.split('Date: ')[1]
        edate = edate.split('\n')[0]
        
        sender = content.split('Return-Path: <')[1] 
        sender = sender.split('>')[0]
        sender.split()

        
        l = 9013
        onehotencoder = one_hot_encode(cbody,l)
        vectorize = vectorize_sequences(onehotencoder,l)
        
        
        endpoint_name = os.environ['ENDPOINT']
        runtime = boto3.Session().client(service_name='sagemaker-runtime',region_name='us-east-1')
        
        payload = json.dumps(vectorize.tolist())
        
        response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=payload)
        
        result = json.loads(response['Body'].read().decode())
       

        label = result.get("predicted_label")
        label = int(label[0][0])
        
        score = result.get("predicted_probability")
        score = score[0][0]*100
        score = round(score,2)
       
        print("Class: ", label)
        print("Score: ", score)
        
        body = """
        We received your email sent at {}  with the
        subject {}.
        Here is a 240 character sample of the email body:
        {} .
        The email was categorized as {} with a {}% confidence
        """.format(edate,subject,cbody,label,score)
        
        message = {"Subject" : {"Data" : subject}, "Body" : {"Html":{"Data": body}}}
        
        resp = ses.send_email(Source = os.environ['SENDER'], Destination = {"ToAddresses":[sender]},Message = message)
        
        print("Sent")
    
    return "Done"