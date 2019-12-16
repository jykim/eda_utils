from setuptools import setup

# def readme():
#     with open('README.md') as f:
#         return f.read()

setup(name='edatools',
      version='0.0.6',
      description='Exploratory Data Analysis Toolkit',
      # long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
      ],
      keywords='Exploratory Data Analysis',
      url='https://github.com/jykim/edatools',
      author='Jin Young Kim',
      author_email='lifidea@gmail.com',
      license='MIT',
      packages=['edatools'],
      install_requires=[
          'numpy', 'pandas', 'bokeh', 'scipy', 'seaborn'
      ],
      include_package_data=True,
      zip_safe=False)