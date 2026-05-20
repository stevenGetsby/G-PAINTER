import torch
import numpy as np
from ....utils.general_utils import dict_foreach
from ....pipelines import samplers


class ClassifierFreeGuidanceMixin:
    def __init__(self, *args, p_uncond: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_uncond = p_uncond

    def get_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"

        if self.p_uncond > 0:
            # randomly drop the class label
            def get_batch_size(cond):
                if isinstance(cond, torch.Tensor):
                    return cond.shape[0]
                elif isinstance(cond, list):
                    return len(cond)
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            ref_cond = cond if not isinstance(cond, dict) else cond[list(cond.keys())[0]]
            B = get_batch_size(ref_cond)

            def select(cond, neg_cond, mask):
                if isinstance(cond, torch.Tensor):
                    mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
                    return torch.where(mask, neg_cond, cond)
                elif isinstance(cond, list):
                    return [nc if m else c for c, nc, m in zip(cond, neg_cond, mask)]
                else:
                    raise ValueError(f"Unsupported type of cond: {type(cond)}")

            mask = list(np.random.rand(B) < self.p_uncond)
            if not isinstance(cond, dict):
                cond = select(cond, neg_cond, mask)
            else:
                cond = dict_foreach([cond, neg_cond], lambda x: select(x[0], x[1], mask))

        return cond


    def get_mixin_cond(self, img_cond, txt_cond, neg_img_cond=None, neg_txt_cond=None, **kwargs):
        """
        Get the conditioning data.
        """
        assert (neg_img_cond is not None) and (neg_txt_cond is not None), "neg_*_cond must be provided for classifier-free guidance"

        if self.p_uncond > 0:
            # 生成一次 mask，同步应用到图像与文本条件
            def get_batch_size(x):
                if isinstance(x, torch.Tensor):
                    return x.shape[0]
                elif isinstance(x, list):
                    return len(x)
                elif isinstance(x, dict):
                    first_key = next(iter(x))
                    return get_batch_size(x[first_key])
                else:
                    raise ValueError(f"Unsupported type of cond: {type(x)}")

            def select(pos, neg, mask):
                if isinstance(pos, torch.Tensor):
                    mask_t = torch.tensor(mask, device=pos.device, dtype=torch.bool).reshape(-1, *[1] * (pos.ndim - 1))
                    return torch.where(mask_t, neg, pos)
                elif isinstance(pos, list):
                    return [n if m else p for p, n, m in zip(pos, neg, mask)]
                elif isinstance(pos, dict):
                    return dict_foreach([pos, neg], lambda xs: select(xs[0], xs[1], mask))
                else:
                    raise ValueError(f"Unsupported type of cond: {type(pos)}")

            ref = img_cond if img_cond is not None else txt_cond
            B = get_batch_size(ref)
            mask = list(np.random.rand(B) < self.p_uncond)

            img_cond = select(img_cond, neg_img_cond, mask)
            txt_cond = select(txt_cond, neg_txt_cond, mask)

        return img_cond, txt_cond

    def get_inference_cond(self, cond, neg_cond=None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        assert neg_cond is not None, "neg_cond must be provided for classifier-free guidance"
        return {'cond': cond, 'neg_cond': neg_cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowCfgSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowCfgSampler()