import importlib.util
import os
import platform
import sys

# Determine the operating system
os_name = platform.system().lower()

# Determine the architecture
arch = 'x86' if platform.machine() == 'x86_64' else 'arm' if platform.machine() == 'arm64' else None

# Map the OS and architecture to the corresponding subdirectory
if os_name == 'darwin' and arch == 'x86':
    subdir = 'macos'
elif os_name == 'darwin' and arch == 'arm':
    subdir = 'macos_arm'
else:
    subdir = {'darwin': None, 'linux': 'linux', 'windows': 'win'}.get(os_name)

# If the subdirectory is found, construct the path to pytelepathy.pyd or pytelepathy.so
if subdir:
    current_dir = os.path.dirname(__file__)
    lib_path = os.path.join(current_dir, subdir, 'pytelepathy')
    lib_path = lib_path + ('.pyd' if os_name == 'windows' else '.so')

    # Load the library dynamically
    spec = importlib.util.spec_from_file_location('pytelepathy', lib_path)
    pytelepathy = importlib.util.module_from_spec(spec)
    sys.modules['animated_drawings.streaming.pytelepathy'] = pytelepathy
    spec.loader.exec_module(pytelepathy)
