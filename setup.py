from setuptools import setup, Extension
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


ext_modules=[
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

for module in ext_modules:
    module.cython_directives = {"language_level": 3}


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
        'scikit-image',
        'scikit-learn',
        'setuptools>=18.0',  # >= 18.0 can handle cython
        'scipy>=1.4.1',
        'sympy',
        'sparseba',
        'tqdm',
        'pyyaml>=5.3'
    ],
    ext_modules=ext_modules,
    cmdclass = {'build_ext': CustomBuildExt},
)
