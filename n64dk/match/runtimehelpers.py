"""Compiler runtime helpers recognised by their code.

A compiler calls its own support routines by names the source never writes (``__udivdi3`` for ``/`` on
unsigned 64-bit values).  A draft that uses the operator makes the compiler emit a call to that name, and the
strict comparison then needs the routine's address.  The name cannot state the address the way ``D_80123456``
does, and nothing in the calling function may establish it; what establishes it is the routine itself: its
code, found at that address, is the routine's and no other function's."""
from __future__ import annotations

from typing import Callable

# libgcc2 (GCC 2.7): ``__udivdi3 (n, d)`` is ``return __udivmoddi4 (n, d, (UDItype *) 0)``: a frame, the
# call, a zero fifth argument word in the delay slot, the epilogue.
_UDIVDI3_HEAD = [0x27BDFFE0, 0xAFBF0018]
_UDIVDI3_TAIL = [0xAFA00010, 0x8FBF0018, 0x27BD0020, 0x03E00008, 0]


def is_udivdi3(read: Callable[[int], int | None] | None, address: int) -> bool:
    if read is None:
        return False
    words = [read(address + 4 * k) for k in range(8)]
    return None not in words and words[:2] == _UDIVDI3_HEAD and words[2] >> 26 == 3 and words[3:] == _UDIVDI3_TAIL


# ``__umoddi3 (u, v)``: ``UDItype w; (void) __udivmoddi4 (u, v, &w); return w;``
_UMODDI3 = [0x27BDFFD8, 0x27A20018, 0xAFBF0020, None, 0xAFA20010, 0x8FA20018, 0x8FA3001C, 0x8FBF0020, 0x27BD0028,
            0x03E00008, 0]


def is_umoddi3(read: Callable[[int], int | None] | None, address: int) -> bool:
    if read is None:
        return False
    words = [read(address + 4 * k) for k in range(len(_UMODDI3))]
    return None not in words and words[3] >> 26 == 3 and all(
        want is None or want == word for want, word in zip(_UMODDI3, words))


RECOGNISERS = {"__udivdi3": is_udivdi3, "__umoddi3": is_umoddi3}


def helpers(read: Callable[[int], int | None], starts) -> dict[str, set[int]]:
    """name -> the function starts whose code is that helper."""
    found: dict[str, set[int]] = {}
    for address in starts:
        for name, recognise in RECOGNISERS.items():
            if recognise(read, address):
                found.setdefault(name, set()).add(address)
    return found
