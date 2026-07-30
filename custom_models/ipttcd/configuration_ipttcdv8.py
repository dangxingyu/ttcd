# -*- coding: utf-8 -*-

from custom_models.ipttcd.configuration_ipttcd import IPTTCDConfig


class IPTTCDv8Config(IPTTCDConfig):

    model_type = "ipttcdv8"

    def __init__(
        self,
        ttt_teacher_conv: bool = True,
        ttt_student_conv: bool = True,
        ttt_conv_kernel_size: int = 5,
        ttt_conv_causal: bool = True,
        ttt_teacher_conv_init: str = "zero",
        ttt_student_conv_init: str = "zero",
        ttt_output_use_teacher_conv: bool = True,
        ttt_output_source: str = "teacher",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ttt_teacher_conv = bool(ttt_teacher_conv)
        self.ttt_student_conv = bool(ttt_student_conv)
        self.ttt_conv_kernel_size = int(ttt_conv_kernel_size)
        self.ttt_conv_causal = bool(ttt_conv_causal)
        self.ttt_teacher_conv_init = str(ttt_teacher_conv_init).lower()
        self.ttt_student_conv_init = str(ttt_student_conv_init).lower()
        self.ttt_output_use_teacher_conv = bool(ttt_output_use_teacher_conv)
        self.ttt_output_source = str(ttt_output_source).lower()
        self.ttt_conv = bool(
            getattr(self, "ttt_conv", False) or self.ttt_teacher_conv or self.ttt_student_conv
        )

        if self.ttt_conv_kernel_size <= 0 or self.ttt_conv_kernel_size % 2 == 0:
            raise ValueError("`ttt_conv_kernel_size` must be a positive odd integer.")
        if self.ttt_output_source not in {"teacher", "student"}:
            raise ValueError("`ttt_output_source` must be either 'teacher' or 'student'.")
        for name, value in {
            "ttt_teacher_conv_init": self.ttt_teacher_conv_init,
            "ttt_student_conv_init": self.ttt_student_conv_init,
        }.items():
            if value not in {"identity", "zero", "random"}:
                raise ValueError(f"`{name}` must be one of 'identity', 'zero', or 'random'.")
