from setuptools import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

shortest_path_module = \
Extension('sarangi.short_path', sources=['sarangi/ext/short_path.pyx', 'sarangi/ext/_short_path.c'], 
          include_dirs=[get_include()], extra_compile_args=['-std=c99'])

setup(name='sarangi',
      description='implementation of string method',
      url='https://github.com/fabian-paul/sarangi',
      author='Fabian Paul',
      author_email='fab@physik.tu-berlin.de',
      license='LGPLv3+',
      packages=['sarangi'],
      install_requires=['mdtraj', 'numpy', 'scipy'],
      ext_modules=cythonize([shortest_path_module]),
      zip_safe=False,
      data_files=[('scripts', ['scripts/string_archive.py', 'scripts/rmsd.py', 'scripts/final_pdb.py', 'scripts/mean_pdb.py'])]
      )