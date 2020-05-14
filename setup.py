import os
from setuptools import setup, Extension
from setuptools_rust import RustExtension, build_ext


def sympy_codegen():
    from tadataka.camera import radtan_codegen
    radtan_codegen.generate()


class CustomBuildExt(build_ext):
    def run(self):
        import numpy as np
        self.include_dirs.append(np.get_include())

        sympy_codegen()

        super().run()


cython_ext_modules=[
    Extension(
        "tadataka.camera._radtan",
        sources=["tadataka/camera/_radtan.pyx",
                 "tadataka/camera/_radtan_distort.c",
                 "tadataka/camera/_radtan_distort_jacobian.c"],
        extra_compile_args=["-Wall", "-Ofast"]
    )
]


for module in cython_ext_modules:
    module.cython_directives = {"language_level": 3}


class get_pybind_include(object):
    def __init__(self, user):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


pybind11_include_dirs = [get_pybind_include(False),
                         get_pybind_include(True),
                         "thirdparty/eigen",
                         os.getcwd()]

pybind11_compile_args = [
    "-O3", "-Wall", "-shared", "-fPIC", "-std=c++14", "-mavx", "-mavx2"
]


def make_pybind11_ext_modules(module_sources):
    pybind11_ext_modules = []
    for module_name, sources in module_sources:
        ext = Extension(
            module_name,
            sources=sources,
            include_dirs=pybind11_include_dirs,
            language="c++",
            extra_compile_args=pybind11_compile_args
        )
        pybind11_ext_modules.append(ext)
    return pybind11_ext_modules


pybind11_module_sources = [
    ("tadataka.camera._normalizer",
     ["tadataka/camera/_normalizer.cpp"]),
    ("tadataka._matrix",
     ["tadataka/_matrix.cpp"]),
    ("tadataka._projection",
     ["tadataka/_projection.cpp"]),
    ("tadataka._transform",
     ["tadataka/_transform.cpp", "tadataka/_matrix.cpp"]),
    ("tadataka._triangulation",
     ["tadataka/_triangulation.cpp", "tadataka/_homogeneous.cpp"]),
    ("tadataka._warp",
     ["tadataka/_warp.cpp", "tadataka/_transform.cpp",
      "tadataka/_projection.cpp"]),
    ("tadataka.vo.semi_dense._depth",
     ["tadataka/vo/semi_dense/_depth.cpp", "tadataka/_projection.cpp"]),
    ("tadataka.vo.semi_dense._epipolar",
     ["tadataka/vo/semi_dense/_epipolar.cpp"]),
    ("tadataka.vo.semi_dense._gradient",
     ["tadataka/vo/semi_dense/_gradient.cpp"]),
    ("tadataka.vo.semi_dense._variance",
     ["tadataka/vo/semi_dense/_variance.cpp", "tadataka/_homogeneous.cpp",
      "tadataka/_matrix.cpp", "tadataka/_projection.cpp"]),
    ("tadataka.vo.semi_dense._intensities",
     ["tadataka/vo/semi_dense/_intensities.cpp"]),
]


setup(
    name='tadataka',
    description='Tadataka',
    url='http://github.com/IshitaTakeshi/Tadataka',
    author='Takeshi Ishita',
    author_email='ishitah.takeshi@gmail.com',
    license='Apache 2.0',
    packages=['tadataka'],
    rust_extensions=[
        RustExtension("rust_bindings.homogeneous", debug=False),
        RustExtension("rust_bindings.interpolation", debug=False),
        RustExtension("rust_bindings.transform", debug=False),
        RustExtension("rust_bindings.projection", debug=False),
        RustExtension("rust_bindings.warp", debug=False),
    ],
    setup_requires=["setuptools-rust"],
    install_requires=[
        'autograd',
        'bidict',
        'cython',
        'matplotlib',
        'numba',
        'numpy>=1.17.3',
        # TODO make independent from opencv
        'opencv-python==4.2.0.34',
        'opencv-contrib-python==4.2.0.34',
        'pandas',
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
    ext_modules=(
        cython_ext_modules +
        make_pybind11_ext_modules(pybind11_module_sources)
    ),
    cmdclass={'build_ext': CustomBuildExt},
    zip_safe=False,
)
