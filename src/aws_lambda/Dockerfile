FROM public.ecr.aws/lambda/python:3.9

RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

COPY /model model
COPY index.py index.py

CMD ["index.handler"]