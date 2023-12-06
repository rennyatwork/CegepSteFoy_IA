from setuptools import setup, find_packages

setup(
    name='regression_model',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        'numpy>=1.18.1,<1.19.0',
        'pandas>=0.25.3,<0.26.0',
        'scikit-learn>=1.3.2,<1.4.0',
        'joblib>=1.1.1,<2.0',
    ],
    entry_points={
        'console_scripts': [
            'regression_model=regression_model.__main__:main',
        ],
    },
)
