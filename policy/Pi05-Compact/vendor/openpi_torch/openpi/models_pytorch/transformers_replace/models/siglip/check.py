import inspect

import transformers

def check_whether_transformers_replace_is_installed_correctly():
    """Check that the OpenPI model replacements are active in Transformers.

    Pi05 relies on both OpenPI's SigLIP mixed-precision implementation and
    its adaptive Gemma RMSNorm.  Checking only the Transformers version lets
    a partially copied replacement pass and fail later at ``cond=...``.
    """
    if transformers.__version__ != "4.53.2":
        return False
    try:
        from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
        from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer, SiglipVisionEmbeddings

        gemma_params = inspect.signature(GemmaRMSNorm.forward).parameters
        gemma_source = inspect.getsource(GemmaRMSNorm.forward)
        if "cond" not in gemma_params or "self.dense.weight.dtype" not in gemma_source:
            return False
        # The replacement contains the explicit LayerNorm dtype boundary
        # required when the Pi05 vision stream runs in bfloat16.
        if "_layer_norm_preserving_dtype" not in inspect.getsource(SiglipEncoderLayer):
            return False
        # Clone the persistent index buffer because DDP broadcasts buffers
        # between TBPTT forwards while earlier graphs are still alive.
        return ".position_ids[:, : embeddings.shape[1]].clone()" in inspect.getsource(SiglipVisionEmbeddings)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
