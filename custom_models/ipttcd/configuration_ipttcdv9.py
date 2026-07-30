# -*- coding: utf-8 -*-

from custom_models.ipttcd.configuration_ipttcdv8 import IPTTCDv8Config


class IPTTCDv9Config(IPTTCDv8Config):

    model_type = "ipttcdv9"

    def __init__(
        self,
        ttt_update_every: int = 4,
        ttt_weight_renorm: bool = False,
        ttt_fuse_proj_into_update: bool = True,
        ttt_force_grouped_scan: bool = False,
        ttt_scan_group_size: int = 4,
        ttt_scan_state_dtype: str = "compute",
        **kwargs,
    ):
        super().__init__(ttt_weight_renorm=ttt_weight_renorm, **kwargs)

        self.ttt_update_every = int(ttt_update_every)
        self.ttt_fuse_proj_into_update = bool(ttt_fuse_proj_into_update)
        self.ttt_force_grouped_scan = bool(ttt_force_grouped_scan)
        self.ttt_scan_group_size = int(ttt_scan_group_size)
        self.ttt_scan_state_dtype = str(ttt_scan_state_dtype).lower()

        if self.ttt_update_every <= 0:
            raise ValueError("`ttt_update_every` must be > 0.")
        if self.ttt_scan_group_size <= 0:
            raise ValueError("`ttt_scan_group_size` must be > 0.")
        if self.ttt_scan_state_dtype not in {"compute", "bf16", "fp32"}:
            raise ValueError(
                "`ttt_scan_state_dtype` must be one of {'compute', 'bf16', 'fp32'}."
            )
