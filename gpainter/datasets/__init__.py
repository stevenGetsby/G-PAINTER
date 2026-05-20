import importlib

__attributes = {
    'BlockFeats': 'block_feats',
    'TextConditionedBlockFeats': 'block_feats',
    'ImageConditionedBlockFeats': 'block_feats',
    'PrecomputedImageConditionedBlockFeats': 'block_feats',
    'BlockPBRDense': 'block_pbr_dense',
    'ImageConditionedBlockPBRDense': 'block_pbr_dense',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
   from .block_feats import (
         BlockFeats,
         TextConditionedBlockFeats,
         ImageConditionedBlockFeats,
         PrecomputedImageConditionedBlockFeats,
   )
   from .block_pbr_dense import (
         BlockPBRDense,
         ImageConditionedBlockPBRDense,
   )
