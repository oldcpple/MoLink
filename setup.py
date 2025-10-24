from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line and not line.startswith('#')]

setup(
    name='molink',
    version='0.1.0',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),  # 从 requirements.txt 中读取依赖
    include_package_data=True,
    description='High-performance and cost-efficient distributed LLM serving engine',
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    author='EmNets',
    author_email='',
    url='https://github.com/oldcpple/molink-rebuild',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
