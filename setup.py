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
        'autograd>=1.3',
        'bidict>=0.18.3',
        'Cython>=0.29.14',
        'matplotlib',
        'numba>=0.46.0',
        'numpy>=1.16.4',
        # TODO make independent from opencv
        'opencv-python-headless>=4.2.0',
        'opencv-contrib-python>=4.2.0',
        'scikit-image>=0.15.0',
        'scikit-learn>=0.21.3',
        'scipy>=1.4.1',
        'sympy>=1.4',
        'sparseba==0.0.1'
    ]
)
