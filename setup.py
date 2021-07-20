from setuptools import setup

setup(name='causticpy',
      version='0.3.2',
      description='Caustic mass estimator for astrophysical systems',
      url='http://github.com/giffordw/causticpy',
      author='Dan Gifford (Modified by Lawrence Bilton)',
      author_email='',
      license='',
      install_requires=['numpy','scipy','cosmolopy','matplotlib','astlib'],
      packages=['causticpy'],
      zip_safe=False)
