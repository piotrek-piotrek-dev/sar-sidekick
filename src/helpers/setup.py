"""
this script is for setting up your environment to implement the DeepSort algorithm for person detection using open Vino
credits and sources from: https://docs.openvino.ai/2024/notebooks/person-tracking-with-output.html#select-inference-device
"""

import subprocess
import sys
from pathlib import Path
from importlib import import_module


PACKAGES_TO_INSTALL = [
    "openvino>=2024.0.0",
    "opencv-python",
    "requests",
    'ultralytics',
]

RAW_FILES_WEB_LOCATIONS = r"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"

MODULES_TO_IMPORT = [
    'collections',
    'datetime',
    'cv2',
    'IPython',
    'pathlib',
    'deepsort_utils.tracker',
    'deepsort_utils.nn_matching',
    'deepsort_utils.detection',
]

NAMESPACES = [
    ('np' , 'numpy'),
    ('ov', 'openvino'),
    ]


SUBMODULES_TO_IMPORT = {}
"""
    'IPython':['display'],
    'pathlib':['Path'],
    'deepsort_utils.tracker':['Tracker'],
    'deepsort_utils.nn_matching':['NearestNeighborDistanceMetric'],
    'deepsort_utils.detection':['Detection',
                                'compute_color_for_labels',
                                'xywh_to_xyxy',
                                'xywh_to_tlwh',
                                'tlwh_to_xyxy'],
}"""

# def import_module(module_name: str, submodule_name: str = None, namespace: str = None) -> bool:
#     try:
#         if (spec := importlib.util.find_spec(name=module_name, package=submodule_name)) is not None:
#             # If you choose to perform the actual import ...
#             module = importlib.util.module_from_spec(spec)
#             sys.modules[module_name] = module
#             spec.loader.exec_module(module)
#             print(f"{module_name!r} has been imported")
#         else:
#             print(f"Could not import {module_name}, aborting")
#             return False
#     except ModuleNotFoundError as mnf:
#         print(f"Could not find {module_name}, while importing, aborting\n{mnf.msg}")
#         return False
#     except ValueError as ve:
#         print(f"Could not parse {module_name}, while importing, aborting\n{ve}")
#         return False
#     except ImportError as e:
#         print(f"Could not import {module_name}, while importing, aborting\n{e.msg}")
#         return False

def import_modules() -> bool:
    for module_name in MODULES_TO_IMPORT:
        try:
            lib = import_module(module_name)
        except ModuleNotFoundError:
            print(f"Failed to import {module_name}. see details:\n{sys.exc_info()}")
            return False
        else:
            globals()[module_name] = lib

    for module_name in SUBMODULES_TO_IMPORT:
        for submodule_name in SUBMODULES_TO_IMPORT[module_name]:
            try:
                lib = import_module(submodule_name, package=module_name)
            except:
                print(f"Failed to import {submodule_name} from {module_name}. see details:\n{sys.exc_info()}")
                return False
            else:
                globals()[f"{module_name}.{submodule_name}"] = lib

    for (namespace, module) in NAMESPACES:
        try:
            lib = import_module(module)
        except:
            print(f"Failed to import {module} as {namespace}. see details:\n{sys.exc_info()}")
            return False
        else:
            globals()[namespace] = lib

    import importlib
    importlib.invalidate_caches()
    return True


def is_package_installed(packageName: str) -> bool:
    if packageName in sys.modules:
        print(f"package {packageName} already installed\n")
        return True
    else:
        print(f"package {packageName} not installed, attempting to install\n")
        return False

def install_notebook() -> bool:
    if not Path("./notebook_utils.py").exists():
        #Fetch `notebook_utils` module
        import requests
        r = requests.get(
            url=RAW_FILES_WEB_LOCATIONS,
        )
        if r.status_code != 200: return False

        with open("notebook_utils.py", "w") as f: f.write(r.text)
        r.close()
        return True

def install_dependencies() -> bool:
    for package in PACKAGES_TO_INSTALL:
        if not is_package_installed(package):
            if not install_package(package): return False
    return True

def install_package(packageName: str) -> bool:
    try:
        print(f"Installing {packageName}...\n")
        ret = subprocess.check_call([sys.executable, "-m", "pip", "install", packageName])
        if ret == 0:
            print(f"package {packageName} is installed\n")
            return True
    except subprocess.CalledProcessError as e:
        print(f"something went wrong with installing {packageName}\n"
              f"check output: {e.output}")
        return False


if __name__ == '__main__':
    if not install_dependencies():
        sys.exit(1)
    if not import_modules():
        sys.exit(1)
