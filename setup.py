from setuptools import setup, find_packages


setup(name='TranReID',
      version='1.0.0',
      description='TransReID: Transformer-based Object Re-Identification',
      author='xxx',
      author_email='xxx',
      url='xxx',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'h5py', 'opencv-python', 'yacs', 'timm==0.3.2'
          ],
      packages=find_packages(),
      keywords=[
          'Pure Transformer',
          'Object Re-identification'
      ])
