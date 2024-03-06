from setuptools import setup, find_packages

setup(
    name='medical_image_segmentation',
    version='0.1.0',
    license='MIT',
    packages=find_packages(),
    description='',
    author='Ethan Haque',
    author_email='ethanhaque@princeton.edu',
    url='https://github.com/EthanHaque/medical-image-segmentation',
    install_requires=[
        "pydicom",
        "numpy"
    ],
    classifiers=[
    ],
)
