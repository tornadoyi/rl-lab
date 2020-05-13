from os.path import dirname, join
from setuptools import setup, find_packages

# Project name
NAME = 'rl-lab'

# Define version information
with open(join(dirname(__file__), NAME + '/VERSION'), 'rb') as f:
      VERSION = f.read().decode('ascii').strip()




setup(name=NAME,
      version=VERSION,
      description="Laboratory of reinforcement learning includes games and algorithms.",
      author='yi gu',
      author_email='390512308@qq.com',
      license='License :: OSI Approved :: Apache Software License',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      python_requires='>=3.7',
      install_requires = [
            'argparse',
            'easydict'
      ],
      entry_points={
            'console_scripts': [
                  'rl-lab = rllab.cli:main',
            ],
      },
)