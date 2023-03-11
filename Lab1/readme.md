### Create two containers with the names "mycontainer1" and "mycontainer2" using the Nginx and MySQL images:

```
docker run -d --name mycontainer1 nginx:latest
docker run -d --name mycontainer2 mysql:latest
docker run -d --name mycontainer3 redis:latest
```

### List all running containers:

```
docker ps

```

### Stop the container with the name "mycontainer1":

```
docker stop mycontainer1

```


### Remove the container with the name "mycontainer1":

```
docker rm mycontainer1


```