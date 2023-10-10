#
FROM python:3.9

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#
COPY ./app /code/app

#
CMD ["hypercorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
# hypercorn app.main:app - log-level debug - reload - bind 127.0.0.1:8091
