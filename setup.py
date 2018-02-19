from setuptools import setup, find_packages

setup(
    name='mbot',
    version='0.1',
    description='Personal Chatbot',
    author='Elisha Yadgaran',
    author_email='ElishaY@alum.mit.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=[
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose']
)
