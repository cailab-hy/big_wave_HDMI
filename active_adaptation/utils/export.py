import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase


@torch.inference_mode()
def export_onnx(module: ModBase, td: TensorDictBase, path: str, meta=None):
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()
    print(torch.__version__)

    # Workaround: onnxscript's version converter crashes on models with functions.
    # The export itself succeeds; patch convert_version to skip on failure.
    _patched = False
    try:
        import onnxscript._framework_apis.torch_2_6 as _onnxscript_api
        _orig_convert = _onnxscript_api.convert_version
        def _safe_convert(model, target_version):
            try:
                return _orig_convert(model, target_version)
            except Exception as e:
                print(f"Warning: ONNX version conversion skipped ({e})")
                return model  # return model unchanged instead of None
        _onnxscript_api.convert_version = _safe_convert
        _patched = True
    except (ImportError, AttributeError):
        pass

    try:
        onnx_program = torch.onnx.dynamo_export(module, **td.to_dict())
    finally:
        if _patched:
            _onnxscript_api.convert_version = _orig_convert

    onnx_program.save(path)
    print(f"Exported ONNX model to {path}.")

    import json

    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = module.in_keys
    meta["out_keys"] = module.out_keys
    meta["in_shapes"] = ([td[k].shape for k in module.in_keys],)

    json.dump(meta, open(meta_path, "w"), indent=4)
    print(f"Exported metadata to {meta_path}.")

    import onnxruntime as ort

    ort_session = ort.InferenceSession(
        path.replace(".pt", ".onnx"), providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnx_input = tuple(td[k] for k in module.in_keys)
    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    ort_output = ort_session.run(None, onnxruntime_input)
    assert len(ort_output) == len(module.out_keys)
