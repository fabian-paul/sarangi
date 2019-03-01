from setuptools import setup


setup(name='sarangi',
      description='implementation of string method',
      url='https://github.com/fabian-paul/sarangi',
      author='Fabian Paul',
      author_email='fab@physik.tu-berlin.de',
      license='LGPLv3+',
      packages=['sarangi'],
      install_requires=['mdtraj'],
      zip_safe=False,
      data_files=[('scripts', ['scripts/rmsd.py', 'scripts/final_pdb.py', 'scripts/mean_pdb.py'])]
      )