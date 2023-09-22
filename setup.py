from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.2'
DESCRIPTION = 'Enables real-world data collection, bridges the gap between OCR and NLP, enabling you to convert text from any image to ready to use nlp data structures.'
LONG_DESCRIPTION = 'Enables real-world data collection, bridges the gap between OCR and NLP, enabling you to convert text from any image to ready to use nlp data structures.'

# Setting up
setup(
    name="pic2prose",
    version=VERSION,
    author="Rohit Mishra",
    author_email="<rohitnmishra2@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['easyocr', 'numpy'],
    keywords=['python', 'images', 'text', 'nlp', 'natural', 'language', 'lexicon'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)