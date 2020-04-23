from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def sympy_codegen():
    from tadataka.camera import radtan_codegen
    radtan_codegen.generate()


class CustomBuildExt(build_ext):
    def run(self):
        sympy_codegen()

        import numpy as np
        self.include_dirs.append(np.get_include())

        build_ext.run(self)


class get_pybind_include(object):
    def __init__(self, user):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


cython_ext_modules=[
    Extension(
        "tadataka.camera._radtan",
        sources=["tadataka/camera/_radtan.pyx",
                 "tadataka/camera/_radtan_distort.c",
                 "tadataka/camera/_radtan_distort_jacobian.c"],
        extra_compile_args=["-Wall", "-Ofast"]
    ),
    Extension(
        "tadataka.interpolation._interpolation",
        sources=["tadataka/interpolation/_interpolation.pyx",
                 "tadataka/interpolation/_bilinear.c"],
        extra_compile_args=["-Wall", "-Ofast", "-mavx", "-mavx2"]
    ),
    Extension(
        "tadataka.vo.semi_dense._intensities",
        sources=["tadataka/vo/semi_dense/_intensities.pyx"],
        extra_compile_args=["-Wall", "-Ofast", "-mavx", "-mavx2"]
    ),
]


for module in cython_ext_modules:
    module.cython_directives = {"language_level": 3}


pybind11_ext_modules = [
    Extension(
        "tadataka.vo.semi_dense._gradient",
        sources=["tadataka/vo/semi_dense/_gradient.cpp"],
        include_dirs=[
            get_pybind_include(False),
            get_pybind_include(True),
            "thirdparty/eigen"
        ],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-shared", "-std=c++11", "-fPIC"],
    ),
]


setup(
    name='tadataka',
    description='Tadataka',
    url='http://github.com/IshitaTakeshi/Tadataka',
    author='Takeshi Ishita',
    author_email='ishitah.takeshi@gmail.com',
    license='Apache 2.0',
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
        'Pillow==7.0.0',
        'pybind11',
        'pyyaml>=5.3',
        'scikit-image',
        'scikit-learn',
        'setuptools>=18.0',  # >= 18.0 can handle cython
        'scipy>=1.4.1',
        'sympy',
        'sparseba',
        'tqdm',
    ],
    ext_modules=cython_ext_modules+pybind11_ext_modules,
    cmdclass = {'build_ext': CustomBuildExt},
)
