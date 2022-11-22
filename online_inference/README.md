online_inference
==============================

2_Homewerok on MLOPS at the MADE

## Usage
1. Create .env file:
~~~
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
BUCKET_TO_MODEL=
PATH_TO_MODEL=
~~~
2. Build image
- From direcory:
~~~
docker build -t dmmaksimov/online_inference:v2 .
~~~
- From [DockerHub](https://hub.docker.com/r/dmmaksimov/online_inference)
~~~
docker pull dmmaksimov/online_inference:v2 .
~~~
3. Run container:
~~~
docker run -d -p 8000:8000 dmmaksimov/online_inference:v2
~~~
5. Run a request scrips:
~~~
PATH_TO_DATA={PATH_TO_DATA} python make_requests.py
~~~
4. Run tests **outside the container**:
~~~
pip install -r requirements.txt
pytest
~~~

## Hostory of Docker image optimizing

1. dmmaksimov/online_inference:v1  
Size 264.42 MB
2. dmmaksimov/online_inference:v2  
--no-cache-dir flag has been added to pip install requirements
Size 163.75 MB
3. dmmaksimov/online_inference:v3  
Layer WORKDIR /online_inference has been deleted
Size 163.75 MB
