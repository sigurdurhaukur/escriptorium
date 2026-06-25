#!/bin/sh


echo "Waiting for postgres..."
while ! nc -z $SQL_HOST $SQL_PORT; do
    sleep 0.1
done
echo "PostgreSQL started"

python manage.py migrate

# Start uwsgi first so nginx doesn't get 502
"$@" &
UWSGI_PID=$!

# Run collectstatic in background while serving
python manage.py collectstatic --no-input

wait $UWSGI_PID
