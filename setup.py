from setuptools import setup, find_packages

setup(
    name='denseav',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'kornia',
        'omegaconf',
        'pytorch-lightning',
        'torchvision',
        'tqdm',
        'torchmetrics',
        'scikit-learn',
        'numpy',
        'matplotlib',
        'timm==0.4.12',
        'moviepy',
        'hydra-core',
        'shutil'
    ],
    author='Mark Hamilton',
    author_email='markth@mit.edu',
    description='Offical code for the CVPR 2024 Paper: Separating the "Chirp" from the "Chat": Self-supervised Visual Grounding of Sound and Language',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mhamilton723/DenseAV',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)
