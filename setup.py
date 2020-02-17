from setuptools import setup, find_packages


setup(
    name='tadataka',
    description='VITAMIN-E',
    url='http://github.com/IshitaTakeshi/VITAMIN-E',
    author='Takeshi Ishita',
    author_email='ishitah.takeshi@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'autograd',
        'bidict',
        'Cython',
        'matplotlib',
        'numba',
        'numpy',
        # TODO make independent from opencv
        'opencv-python',
        'opencv-contrib-python',
        'pandas',
        'scikit-image',
        'scikit-learn',
        'scipy>=1.4.1',
        'sympy',
        'sparseba',
        'pyyaml>=5.3'
    ]
)
