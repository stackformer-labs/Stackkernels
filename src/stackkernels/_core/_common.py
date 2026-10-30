"""cdiv, next_pow2, constexpr enums"""

import triton

cdiv = triton.cdiv
next_pow2 = triton.next_power_of_2

__all__ = ["cdiv", "next_pow2"]
