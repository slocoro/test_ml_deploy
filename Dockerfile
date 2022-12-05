FROM python:3.9-slim

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . ./opt/code

WORKDIR ./opt/code

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "80"]