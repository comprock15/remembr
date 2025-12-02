# ReMEmbR Module

## Setup

1. Download VILA

```
mkdir deps
cd deps
git clone https://github.com/NVlabs/VILA.git
cd VILA
git checkout c8f603b49f5dcfca8c2ee18d7979897a83aa5fa6
```

2. Build the container

```
cd ../../remembr
bash docker/build.sh
```

3. Install MilvusDB

```
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o launch_milvus_container.sh
 
bash launch_milvus_container.sh start
```

4. Run the container

```
bash docker/start.sh
```

5. Download llama3.1:8b
```
conda activate remembr
ollama pull llama3.1:8b
```

## Launch HTTPS-wrapper

```
bash docker/into.sh
cd /home/docker_user/remembr
```

```
export PYTHONPATH="/home/docker_user/remembr/deps/VILA::/home/docker_user/remembr"
```

```
cd remembr
```

```
conda activate remembr
```

```
python scripts/remembr_api.py 
```

Now you can send requests to ReMEmbR module!
