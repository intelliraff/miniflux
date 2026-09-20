"""Load pinned Meta reference modules without changing their model computation.

Only the absolute models.nn import is redirected to an isolated namespace.
Original files and license remain in third_party/flow_matching.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT/'third_party/flow_matching'
COMMIT = '11568d37f8d5a080e12aa7b5305d9c35ae07d136'


def module(name, path, redirect=False):
    if name in sys.modules: return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    if redirect:
        source=path.read_text()
        assert source.count('from models.nn import (')==1
        source=source.replace('from models.nn import (','from _meta_cifar.nn import (')
        exec(compile(source,str(path),'exec'),result.__dict__)
    else: spec.loader.exec_module(result)
    return result


def reference_components():
    if not REFERENCE.exists(): raise FileNotFoundError('Clone the pinned Meta reference first; see experiments/CIFAR10.md')
    package=sys.modules.setdefault('_meta_cifar',types.ModuleType('_meta_cifar'))
    package.__path__=[str(REFERENCE/'examples/image/models')]
    module('_meta_cifar.nn',REFERENCE/'examples/image/models/nn.py')
    unet=module('_meta_cifar.unet',REFERENCE/'examples/image/models/unet.py',redirect=True)
    ema=module('_meta_cifar.ema',REFERENCE/'examples/image/models/ema.py')
    schedule=module('_meta_cifar.schedule',REFERENCE/'examples/image/training/edm_time_discretization.py')
    tree=ast.parse((REFERENCE/'examples/image/models/model_configs.py').read_text())
    configs=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODEL_CONFIGS' for t in n.targets))
    if str(REFERENCE) not in sys.path: sys.path.insert(0,str(REFERENCE))
    from flow_matching.path import CondOTProbPath
    return unet.UNetModel, configs['cifar10'], ema.EMA, schedule.get_time_discretization, CondOTProbPath


def make_reference():
    cls,config,*_=reference_components()
    return cls(**config)
