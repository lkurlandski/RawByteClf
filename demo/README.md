# 

To create the docker image:
```
sudo docker build -t demo:latest .
sudo docker save demo:latest | gzip > demo.tar.gz
```

To use the docker image:
```
gunzip -c demo.tar.gz | sudo docker load
sudo docker run --rm demo:latest
```

To clean up docker artifacts:
```
sudo docker container prune -f
sudo docker image prune -f
docker system prune -a --volumes -f
```
