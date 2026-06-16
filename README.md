# AORC-Heat-Pipeline

To run unit tests on modified source code:
```
git clone https://github.com/PersadAeroClimateLab/AORC-Heat-Pipeline
cd AORC-Heat-Pipeline
docker build -t aorc .
docker run -v .:/project -it aorc pytest tests/
```

To run the pipeline, either load and environment with the dependencies in `requirements.txt` and execute `run.py` via
```
python run.py
```
or use the Docker container (after building with `docker build` above):
```
docker run -it aorc python run.py
```