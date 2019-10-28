from setuptools import setup, find_packages


setup(
    name='vitamine',
    description='VITAMIN-E',
    url='http://github.com/IshitaTakeshi/VITAMIN-E',
    author='Takeshi Ishita',
    author_email='ishitah.takeshi@gmail.com',
    license='MIT',
    packages=['vitamine'],
    install_requires=[
        'autograd',
        'scipy',
        'scikit-image',
        'matplotlib',
        'numpy',
        # TODO make independent from opencv
        'opencv-python',
        'opencv-contrib-python'
    ],
    dependency_links=[
        'git+https://github.com/IshitaTakeshi/SBA.git@develop#egg=sba'
    ]
)
