FROM datmo/data-analytics:cpu-py35

RUN pip install sagemaker boto3

RUN apt-get update -y && apt-get install -y unzip

RUN curl "https://d1vvhvl2y92vvt.cloudfront.net/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install

