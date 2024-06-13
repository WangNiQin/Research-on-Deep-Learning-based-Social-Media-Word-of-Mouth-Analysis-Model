from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
  name = 'vit-pytorch',
  packages = find_packages(exclude=['examples']),
  version = '1.7.0',
  description = 'Adapt-Vision Transformer (AViT) - Pytorch',
  long_description=long_description,
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'image recognition'
  ],
  install_requires=[
    'einops>=0.7.0',
    'torch>=1.10',
    'torchvision'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest',
    'torch==1.12.1',
    'torchvision==0.13.1'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Programming Language :: Python :: 3.7',
  ],
)
