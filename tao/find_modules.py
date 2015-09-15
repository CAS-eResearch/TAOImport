import os, importlib, inspect
from Module import Module

def add_module(module, ordered_modules):
    if hasattr(module, 'dependencies'):
        for dep in module.dependencies:
            add_module(dep, ordered_modules)
    if module not in ordered_modules:
        ordered_modules.append(module)

def find_modules():
    modules = []
    dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
    for entry in os.listdir(dir):
        path = os.path.join(dir, entry)
        if os.path.isfile(path) and entry[-3:].lower() == '.py':
            modules.extend(import_modules(entry[:-3]))
    ordered_modules = []
    for mod in modules:
        add_module(mod, ordered_modules)
    return ordered_modules

def import_modules(entry):
    modules = []
    mod = importlib.import_module('tao.' + entry)
    for name, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and obj != Module and issubclass(obj, Module):
            modules.append(obj)
    return modules
