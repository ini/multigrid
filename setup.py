import pathlib
import setuptools



PACKAGE_DIR = pathlib.Path(__file__).absolute().parent



def get_version():
    """
    Gets the multigrid version.
    """
    path = PACKAGE_DIR / 'multigrid' / '__init__.py'
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith('__version__'):
            return line.strip().split()[-1].strip().strip("'")

    raise RuntimeError("bad version data in __init__.py")

def get_description():
    """
    Gets the description from the readme.
    """
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith('##'):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return long_description

setuptools.setup(
    name='multigrid',
    version=get_version(),
    long_description=get_description(),
)
