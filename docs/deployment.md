## Deployment with SSL

If you would like to deploy with SSL, follow these steps:

Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365

Start gunicorn with uvicorn worker thread and SSL
gunicorn -b 0.0.0.0:5000 --keyfile=/tmp/key.pem --certfile=/tmp/cert.pem -k uvicorn.workers.UvicornWorker -w 12 app:app

Start uvicorn with reload and SSL
uvicorn --host 0.0.0.0 --port 5000 --ssl-keyfile=/tmp/key.pem --ssl-certfile=/tmp/cert.pem  --reload app:app
