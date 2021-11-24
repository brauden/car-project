install:
  conda env create -f environment.yml

lint:
  pylint ./models && pylint ./servers
  
test:
  pytest ./tests/tests.py
