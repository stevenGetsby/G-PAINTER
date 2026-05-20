import importlib

__attributes = {
    'OctreeRenderer': 'octree_renderer',
    'GaussianRenderer': 'gaussian_render',
    'MeshRenderer': 'mesh_renderer',
    'render_random_normals_grid': 'normal_render',
    'simplify_mesh_gpu': 'normal_render',
    'load_and_simplify': 'normal_render',
    'render_pbr_views': 'pbr_renderer',
    'render_pbr_grid': 'pbr_renderer',
    'render_block_pbr_views': 'pbr_renderer',
    'render_block_pbr_shaded': 'pbr_renderer',
    'render_mesh_textured': 'pbr_renderer',
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
    from .octree_renderer import OctreeRenderer
    from .gaussian_render import GaussianRenderer
    from .mesh_renderer import MeshRenderer