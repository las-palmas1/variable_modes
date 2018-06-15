from distutils.core import setup

setup(
    name='variable_modes',
    version='0.0.1',
    package_dir={'variable_modes': 'core',
                 'variable_modes.compressor_characteristics': 'core/compressor_characteristics'},
    packages=['variable_modes', 'variable_modes.compressor_characteristics'],
    url='',
    license='',
    author='Alexander Zhigalkin',
    author_email='aszhigalkin94@gmail.com',
    description='Library for computing variable modes of gas turbine'
)
