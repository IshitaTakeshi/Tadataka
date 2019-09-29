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
        # 'julia',
        'opencv-python'  # TODO make independent from this
    ]
)
