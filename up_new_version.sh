python3 -m pip install --upgrade setuptools twine
rm -rf dist/*
python3 setup.py sdist
twine upload dist/*