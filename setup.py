from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


def sympy_codegen():
    from tadataka.camera import radtan_codegen
    radtan_codegen.generate()


class CustomBuildExt(build_ext):
    def run(self):
        import numpy as np

        sympy_codegen()

        self.include_dirs.append(np.get_include())
        build_ext.run(self)


setup(
    name='tadataka',
    description='VITAMIN-E',
    url='http://github.com/IshitaTakeshi/VITAMIN-E',
    author='Takeshi Ishita',
    author_email='ishitah.takeshi@gmail.com',
    license='MIT',
    packages=['tadataka'],
    install_requires=[
        'autograd',
        'bidict',
        'cython',
        'matplotlib',
        'numba',
        'numpy',
        # TODO make independent from opencv
        'opencv-python',
        'opencv-contrib-python',
        'pandas',
        'scikit-image',
        'scikit-learn',
        'setuptools>=18.0',  # >= 18.0 can handle cython
        'scipy>=1.4.1',
        'sympy',
        'sparseba',
        'pyyaml>=5.3'
    ],
    ext_modules=[
        Extension('tadataka.camera._radtan',
                  sources=["tadataka/camera/_radtan.pyx",
                           "tadataka/camera/_radtan_distort_jacobian.c"]),
    ],
    cmdclass = {'build_ext': CustomBuildExt},

)
