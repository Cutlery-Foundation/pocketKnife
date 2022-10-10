# pocketKnife

- A collection of useful functions for any data scientist working with NLP.
- Containes TF-IDF, embedders, and data cleaning functions


### reference
- https://towardsdatascience.com/create-your-own-python-package-and-publish-it-into-pypi-9306a29bc116




```bash
# install requirements
python -m pip install --upgrade setuptools twine

# upgrade pip version [GUI]
edit setup.py file line <version='0.x.x'>

# delete old dists [CLI]
rm -rf dist/*

# create new dist [CLI]
python setup.py sdist

# upload new dist [CLI]
twine upload dist/*

# commit new dist [CLI]
git commit -am "0.0.9"
git push
```