import setuptools

setuptools.setup(
    name='pocketKnife',
    version='0.0.17',
    maintainer='Cutlery Foundation',
    maintainer_email='cutlery.foundation@gmail.com',
    description='A collection of useful functions for any data scientist working with NLP',
    packages=['pocketKnife'],
    url='https://github.com/Cutlery-Foundation/PocketKnife',
    install_requires=[
        'pandas>=1.3.5', #1.5.0
        'matplotlib',
        'plotly',
        'scikit-learn',
        'scikit-multilearn',
        'pyod',
        'lightgbm',
        'xgboost',
        'catboost',
        # 'ipython',
        'ipywidgets==8.0.2',
        'ipympl',
        'scipy>=1.7.3', #1.9.1
        'wordcloud==1.8.2.2',
        'black==22.10.0',
        'beautifulsoup4==4.11.1',
        'lxml==4.9.1',
        'tqdm==4.64.1',
        'sentence_transformers==2.2.2',
        'setuptools==65.4.1',
        'wheel==0.37.1',
        'spacy==3.4.1',
        # 'ipykernel==6.16.0',
        'seaborn==0.12.0',
        'pyarrow==9.0.0',
        'unidecode',
        'py3nvml'
    ],
    extras_require={
        'cuda': [
            'torch',
            'torchvision',
            'torchaudio',
            'spacy[transformers,lookups,cuda117]==3.4.1',
        ]
    },
    dependency_links=['https://download.pytorch.org/whl/cu117',]
)