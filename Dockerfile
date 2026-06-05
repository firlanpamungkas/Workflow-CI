FROM python:3.12-slim

WORKDIR /app

COPY MLProject/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "MLProject/modelling.py"]