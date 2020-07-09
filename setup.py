from setuptools import setup

# def readme():
#     with open('README.md') as f:
#         return f.read()

setup(name='e3tools',
      version='0.0.6',
      description='Data Science Toolkit from E3Data',
      # long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
      ],
      keywords='Exploratory Data Analysis',
      url='https://github.com/jykim/e3tools',
      author='Jin Young Kim',
      author_email='lifidea@gmail.com',
      license='MIT',
      packages=['e3tools'],
      install_requires=[
          'numpy', 'pandas', 'bokeh', 'scipy', 'seaborn', 'sklearn', 'imblearn', 'impyute'
      ],
      include_package_data=True,
      zip_safe=False)