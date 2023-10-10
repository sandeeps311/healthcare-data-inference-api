#
FROM python:3.9

#
WORKDIR /code

#
COPY requirements.txt /code/requirements.txt

#
RUN pip install -r /code/requirements.txt
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY . /code/app

#
CMD ["hypercorn", "src.main:app", "--bind", "0.0.0.0:8080"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
# hypercorn app.main:app - log-level debug - reload - bind 127.0.0.1:8091
