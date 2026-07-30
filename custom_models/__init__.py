from importlib import import_module


def _safe_import(module_name: str) -> None:
    try:
        import_module(module_name)
    except Exception:
        # Keep custom model registration best-effort so environments that only
        # support a subset of optional model dependencies can still boot.
        pass


_safe_import("custom_models.sba")
_safe_import("custom_models.universal_transformer")
_safe_import("custom_models.loop_transformer")
_safe_import("custom_models.kimi_linear")
_safe_import("custom_models.moe_transformer")
_safe_import("custom_models.moe_loop_transformer")
_safe_import("custom_models.mole")
_safe_import("custom_models.mole_loop")
_safe_import("custom_models.mole_parscale")
_safe_import("custom_models.parscale")
_safe_import("custom_models.ipttt")
_safe_import("custom_models.ipttcd")
