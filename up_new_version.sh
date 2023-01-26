python3 -m pip install --upgrade setuptools twine pip setuptools wheel
rm -rf dist/*
python3 setup.py sdist
twine upload dist/*