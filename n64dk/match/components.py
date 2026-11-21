"""Toolchain components and build recipes as a graph.

A label such as ``ido5.3`` or ``kmc+as`` compresses a whole build into one word: preprocessing,
the language front end, the optimiser, the code generator, the assembler with its macro dialect
and scheduling, the object format and the conversions after it.  A mismatch can come from any of
them, and two labels can share all but one.  This module names each stage's component, where it
comes from and what is known about it, and describes every lane the kit drives as a *recipe*: the
ordered stages, the component of each, the flags each stage receives, and a stable identity
built from the components' file digests.

Provenance classes (what a component *is*), kept apart on purpose:

``authentic-n64``             the tool that built retail N64 code, as shipped (possibly run under
                              static recompilation or an emulator: the code is the original's)
``authentic-n64-component``   an authentic N64 tool used for one stage in a combination its
                              vendor did not ship (e.g. KMC cc1 with another assembler)
``authentic-assembler-linker`` an original assembler, linker or object tool of an N64 kit
``reconstructed``             a rebuild or reimplementation meant to behave like an authentic
                              tool (rebuilt GCC sources, an assembler converter, an object parser)
``comparative``               a relative from another platform or a homebrew compiler: a research
                              source and a negative control, never an N64 lane by itself
``speculation``               named in discussion without artefact or evidence

Dispositions (where an investigated item ends): ``implemented-verified``, ``implemented-unvalidated``
(built and usable, but that it is what a retail build used is a hypothesis), ``comparative-only``,
``rejected``, ``unavailable``, ``undistributable-local``, ``speculation``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROLES = ("driver", "preprocessor", "frontend", "optimizer", "codegen", "compiler", "assembler-macro",
         "assembler", "object-converter", "linker", "archiver", "post-link", "emulator", "library")
PROVENANCE = ("authentic-n64", "authentic-n64-component", "authentic-assembler-linker", "reconstructed",
              "comparative", "speculation")
DISPOSITIONS = ("implemented-verified", "implemented-unvalidated", "comparative-only", "rejected", "unavailable",
                "undistributable-local", "speculation")


@dataclass(frozen=True)
class Component:
    ident: str
    role: str
    name: str
    version: str
    provenance: str
    disposition: str
    origin: str                          # where the artefact comes from
    source_url: str = ""
    files: tuple[str, ...] = ()          # paths inside the toolchain directory that make it up
    directory: str = ""                  # toolchain directory (under the toolchain root)
    platform: str = ""                   # executable format / host it runs on
    emulator: str = ""                   # compatibility layer it needs, if any
    target: str = "mips-n64 (big-endian, o32)"
    language: str = "C"
    licence: str = ""
    known_projects: str = ""
    known_flags: tuple[str, ...] = ()
    retail_evidence: str = ""
    confidence: str = "medium"
    notes: str = ""


def _c(*args, **kwargs) -> Component:
    return Component(*args, **kwargs)


_DECOMPME = "decomp.me compiler build definitions (github.com/decompme/compilers values.yaml, read 2026-11-16)"
_IDO_RECOMP = "https://github.com/decompals/ido-static-recomp (v1.2)"
_IDO_NOTE = ("IRIX binaries translated to native code by static recompilation; behaviour is the original "
             "binary's, which matching projects verify byte for byte")

COMPONENTS: tuple[Component, ...] = (
    # ---- SGI IDO 5.3 (static recompilation)
    _c("ido5.3/cc", "driver", "IDO 5.3 cc driver", "5.3", "authentic-n64", "implemented-verified", _DECOMPME,
       _IDO_RECOMP, ("cc",), "ido5.3", "x86-64 ELF (recompiled from IRIX MIPS)", licence="SGI proprietary; "
       "redistributed by the matching community", known_projects="SM64, OoT, MM, Banjo-Kazooie, libultra <= 2.0I",
       known_flags=("-O1", "-O2", "-g", "-mips1", "-mips2", "-G 0", "-non_shared", "-Xcpluscomm", "-Wab,-r4300_mul"),
       retail_evidence="whole titles match strictly (SM64: 3,205 functions this round)", confidence="high",
       notes=_IDO_NOTE),
    _c("ido5.3/cfe", "frontend", "IDO 5.3 C front end (with its built-in preprocessor)", "5.3", "authentic-n64",
       "implemented-verified", _DECOMPME, _IDO_RECOMP, ("cfe",), "ido5.3", "x86-64 ELF (recompiled)",
       notes="emits ucode; preprocesses internally unless -acpp; line numbers travel in the ucode (-g)"),
    _c("ido5.3/acpp", "preprocessor", "IDO 5.3 ANSI preprocessor", "5.3", "authentic-n64", "implemented-verified",
       _DECOMPME, _IDO_RECOMP, ("acpp",), "ido5.3", "x86-64 ELF (recompiled)", notes="used with -acpp"),
    _c("ido5.3/uopt", "optimizer", "IDO 5.3 global optimiser", "5.3", "authentic-n64", "implemented-verified",
       _DECOMPME, _IDO_RECOMP, ("uopt",), "ido5.3", "x86-64 ELF (recompiled)", notes="runs at -O1 and above"),
    _c("ido5.3/ugen", "codegen", "IDO 5.3 code generator", "5.3", "authentic-n64", "implemented-verified",
       _DECOMPME, _IDO_RECOMP, ("ugen",), "ido5.3", "x86-64 ELF (recompiled)"),
    _c("ido5.3/as1", "assembler", "IDO 5.3 assembler pass 1 (scheduling, delay slots)", "5.3", "authentic-n64",
       "implemented-verified", _DECOMPME, _IDO_RECOMP, ("as1",), "ido5.3", "x86-64 ELF (recompiled)",
       notes="fills delay slots and schedules; encodes move as or; as0 expands assembler macros"),
    _c("ido5.3/as0", "assembler-macro", "IDO 5.3 assembler pass 0 (macro expansion)", "5.3", "authentic-n64",
       "implemented-verified", _DECOMPME, _IDO_RECOMP, ("as0",), "ido5.3", "x86-64 ELF (recompiled)"),
    _c("ido5.3/uld", "optimizer", "IDO 5.3 ucode linker/inliner (-O3 pipeline: uld, umerge, usplit, ujoin)", "5.3",
       "authentic-n64", "implemented-verified", _DECOMPME, _IDO_RECOMP, ("uld", "umerge", "usplit", "ujoin"),
       "ido5.3", "x86-64 ELF (recompiled)", notes="whole-unit -O3 inlining; libultra gu*/al* are built -O3"),
    _c("ido5.3/copt", "optimizer", "IDO 5.3 source optimiser (copt)", "5.3", "authentic-n64", "comparative-only",
       _DECOMPME, _IDO_RECOMP, ("copt",), "ido5.3", "x86-64 ELF (recompiled)",
       notes="-sopt only; no retail evidence of its use found"),
    _c("ido5.3/ld", "linker", "IDO 5.3 ld", "5.3", "authentic-assembler-linker", "comparative-only", _DECOMPME,
       _IDO_RECOMP, ("ld",), "ido5.3", "x86-64 ELF (recompiled)",
       notes="retail N64 builds were linked by the SDK's makerom/spicy or ld scripts; the kit links with GNU ld"),
    # ---- SGI IDO 7.1
    _c("ido7.1/cc", "driver", "IDO 7.1 cc driver", "7.1", "authentic-n64", "implemented-verified", _DECOMPME,
       _IDO_RECOMP, ("cc",), "ido7.1", "x86-64 ELF (recompiled)", known_projects="later first-party titles, "
       "libultra 2.0J+", known_flags=("-O2", "-mips2", "-g"), confidence="high", notes=_IDO_NOTE),
    _c("ido7.1/cfe", "frontend", "IDO 7.1 C front end", "7.1", "authentic-n64", "implemented-verified", _DECOMPME,
       _IDO_RECOMP, ("cfe",), "ido7.1", "x86-64 ELF (recompiled)"),
    _c("ido7.1/uopt", "optimizer", "IDO 7.1 global optimiser", "7.1", "authentic-n64", "implemented-verified",
       _DECOMPME, _IDO_RECOMP, ("uopt",), "ido7.1", "x86-64 ELF (recompiled)"),
    _c("ido7.1/ugen", "codegen", "IDO 7.1 code generator", "7.1", "authentic-n64", "implemented-verified",
       _DECOMPME, _IDO_RECOMP, ("ugen",), "ido7.1", "x86-64 ELF (recompiled)"),
    _c("ido7.1/as1", "assembler", "IDO 7.1 assembler pass 1", "7.1", "authentic-n64", "implemented-verified",
       _DECOMPME, _IDO_RECOMP, ("as1", "as0"), "ido7.1", "x86-64 ELF (recompiled)"),
    _c("ido7.1/edgcpfe", "frontend", "IDO 7.1 EDG C++ front end", "7.1", "authentic-n64", "comparative-only",
       _DECOMPME, _IDO_RECOMP, ("edgcpfe", "NCC"), "ido7.1", "x86-64 ELF (recompiled)", language="C++",
       notes="the kit drafts C; no N64 C++ title validated with it"),
    _c("ido7.1/upas", "frontend", "IDO 7.1 Pascal front end", "7.1", "speculation", "speculation", _DECOMPME,
       _IDO_RECOMP, ("upas",), "ido7.1", "x86-64 ELF (recompiled)", language="Pascal",
       retail_evidence="none found", confidence="low",
       notes="ships with IDO; no retail N64 code is known to have been written in Pascal"),
    # ---- IDO releases under an IRIX user-mode emulator
    _c("ido4.1/cc", "driver", "IDO 4.1 (IRIX 4, ECOFF output)", "4.1", "authentic-n64", "unavailable", _DECOMPME,
       "https://github.com/decompme/compilers (ido4.1.tar.gz)", ("usr/bin/cc",), "ido4.1", "IRIX MIPS ECOFF",
       emulator="IRIX user-mode emulator", known_projects="very early SDK code",
       notes="no lane: its ECOFF objects need a conversion step the kit does not drive"),
    _c("ido5.2/driver", "driver", "IDO 5.2", "5.2", "authentic-n64", "implemented-verified", _DECOMPME,
       "https://github.com/LLONSIT/qemu-irix-helpers (ido5.2.tar.xz)", ("usr/lib/driver",), "ido5.2",
       "IRIX MIPS ELF", emulator="qemu-irix", known_projects="early titles"),
    _c("ido6.0/driver", "driver", "IDO 6.0", "6.0", "authentic-n64", "implemented-verified", _DECOMPME,
       "https://github.com/LLONSIT/qemu-irix-helpers (ido6.0.tar.xz)", ("usr/lib/driver",), "ido6.0",
       "IRIX MIPS ELF", emulator="qemu-irix"),
    _c("mips_pro_744/driver", "driver", "MIPSpro 7.4.4", "7.4.4", "authentic-n64", "implemented-verified",
       _DECOMPME, "https://github.com/LLONSIT/qemu-irix-helpers (mipspro7.4.4.tar.xz)", ("usr/lib/driver",),
       "mips_pro_744", "IRIX MIPS ELF", emulator="qemu-irix", confidence="low", known_projects="rare"),
    _c("ido5.3_cfront/CC", "frontend", "IDO 5.3 C++ (cfront)", "5.3", "authentic-n64", "comparative-only",
       _DECOMPME, "https://github.com/decompme/compilers (ido5.3_cfront.tar.gz)", ("CC",), "ido5.3_cfront",
       "IRIX MIPS ELF", emulator="qemu-irix", language="C++", notes="the kit drafts C"),
    _c("qemu-irix", "emulator", "qemu-irix (IRIX user-mode emulation)", "", "reconstructed", "implemented-verified",
       "bundled with the IRIX compiler archives", "https://github.com/LLONSIT/qemu-irix-helpers",
       ("usr/bin/qemu-irix",), "ido6.0", "x86-64 ELF", notes="runs authentic IRIX binaries; the binaries' "
       "behaviour is the original's, the emulator is not part of the output"),
    # ---- KMC GCC 2.7.2 (the Windows SDK compiler), rebuilt
    _c("kmc/gcc", "driver", "KMC GCC 2.7.2 driver", "2.7.2", "reconstructed", "implemented-verified", _DECOMPME,
       "https://github.com/decompals/mips-gcc-2.7.2", ("gcc",), "kmc-gcc-2.7.2", "x86-64 ELF",
       licence="GPL (GCC)", known_projects="Dr. Mario 64, many third-party titles, libgultra",
       known_flags=("-O2", "-O1", "-G0", "-mips3", "-mips2", "-mgp32", "-mfp32"), confidence="high",
       retail_evidence="Dr. Mario 64: 732 strict functions this round",
       notes="a build of GCC 2.7.2 sources reproducing the KMC compiler's output, verified by matching projects"),
    _c("kmc/cpp", "preprocessor", "KMC GCC 2.7.2 cpp (cccp)", "2.7.2", "reconstructed", "implemented-verified",
       _DECOMPME, "https://github.com/decompals/mips-gcc-2.7.2", ("cpp", "cccp"), "kmc-gcc-2.7.2", "x86-64 ELF",
       licence="GPL"),
    _c("kmc/cc1", "compiler", "KMC GCC 2.7.2 cc1 (front end, optimiser and code generator in one)", "2.7.2",
       "reconstructed", "implemented-verified", _DECOMPME, "https://github.com/decompals/mips-gcc-2.7.2", ("cc1",),
       "kmc-gcc-2.7.2", "x86-64 ELF", licence="GPL",
       notes="monolithic: the three stages cannot be exchanged separately; -g changes line notes, not tokens"),
    _c("kmc/as", "assembler", "KMC GNU as (binutils 2.6)", "2.6", "reconstructed", "implemented-verified",
       _DECOMPME, "https://github.com/decompals/mips-binutils-2.6", ("as",), "kmc-gcc-2.7.2", "x86-64 ELF",
       licence="GPL", notes="reorders and fills delay slots itself (.set reorder); encodes move as addu"),
    _c("gnu/as-2.9", "assembler", "period GNU as 2.9.x (gcc 2.8.1 / egcs 1.1.2 toolchains)", "2.9.1",
       "reconstructed", "implemented-unvalidated", _DECOMPME, "", ("as",), "gcc2.8.1pm", "x86-64 ELF", licence="GPL",
       notes="round 12: unlike KMC as 2.6 it makes a symbol declared '.extern name, size' GP-relative under -G "
             "(probe: bench/analysis gprobe); the kmc+gas lane pairs it with KMC GCC for small-data (-G) code.  "
             "A period GNU assembler rebuilt for a modern host (reconstructed): that a retail build used it is "
             "a hypothesis the strict matches it yields support (round 12: 25 strict exact, 0 lost)"),
    _c("kmc/ar", "archiver", "KMC binutils ar", "2.6", "reconstructed", "comparative-only", _DECOMPME,
       "https://github.com/decompals/mips-binutils-2.6", ("ar",), "kmc-gcc-2.7.2", "x86-64 ELF"),
    _c("kit/asn64-converter", "assembler-macro", "kit ASN64-compatible assembly converter", "round 10",
       "reconstructed", "implemented-verified", "n64dk/match/asn64.py", "", (), "", "Python",
       notes="rewrites KMC cc1 output to the conventions of SN's asn64 before a modern GNU as"),
    _c("kit/at-symbol-loads", "assembler-macro", "kit expansion of symbol loads through $at", "round 11",
       "reconstructed", "implemented-unvalidated", "n64dk/match/normalize.py (at_symbol_loads)", "", (), "", "Python",
       notes="emulates an assembler that expands lw/lh/lb rD,sym through $at (seen in retail code; KMC as 2.6, "
             "GNU as 2.9.1, ASN64 2.56/2.72/2.81 and IRIX as all use rD); offered only to functions whose own "
             "code shows the form"),
    _c("kit/at-branch-macros", "assembler-macro", "kit rewrite of branch comparisons as branch macros", "round 12",
       "reconstructed", "implemented-unvalidated", "n64dk/match/normalize.py (at_branch_macros)", "", (), "", "Python",
       notes="slt/sltu/slti/sltiu into a register that only feeds bne/beq against $0 becomes blt/bge/bltu/bgeu, "
             "which every probed assembler expands through $at (bench/macros); applied only when the register is "
             "dead on both paths; offered only to functions whose own code shows comparisons through $at"),
    _c("binutils/as", "assembler", "GNU as (modern binutils, mips-linux-gnu)", "2.42", "reconstructed",
       "implemented-verified", "distribution package", "", ("usr/bin/mips-linux-gnu-as",), "mips-binutils",
       "x86-64 ELF", licence="GPL", notes="encodes move as or; used with --no-pad-sections behind converters"),
    _c("binutils/ld", "linker", "GNU ld (modern binutils)", "2.42", "reconstructed", "implemented-verified",
       "distribution package", "", ("usr/bin/mips-linux-gnu-ld.bfd",), "mips-binutils", "x86-64 ELF",
       licence="GPL", notes="the kit's linker for donor images and probes; not a retail linker"),
    # ---- SN Systems (N64)
    _c("sn/cc1n64-2.7.2-0001", "compiler", "SN Systems cc1n64 (GCC 2.7.2, build 0001)", "2.7.2 b0001",
       "authentic-n64", "implemented-verified", _DECOMPME,
       "https://github.com/decompme/compilers (n64_sn272_0001.tar.gz)", ("cc1n64.exe",), "gcc2.7.2sn0001",
       "PE32 (Windows)", emulator="wibo", licence="SN Systems proprietary; redistributed by the matching community",
       known_flags=("-O2", "-G0", "-mips3", "-mhard-float", "-meb"), notes="does not know -mcpu=vr4300"),
    _c("sn/cc1n64-2.7.2-0004", "compiler", "SN Systems cc1n64 (GCC 2.7.2, build 0004)", "2.7.2 b0004",
       "authentic-n64", "implemented-verified", _DECOMPME, "https://github.com/Mr-Wiseguy/pcsx-redux (n64 release)",
       ("cc1n64.exe",), "gcc2.7.2sn0004", "PE32 (Windows)", emulator="wibo"),
    _c("sn/cc1n64-2.7.2-0006", "compiler", "SN Systems cc1n64 (GCC 2.7.2, build 0006)", "2.7.2 b0006",
       "authentic-n64", "implemented-verified", _DECOMPME,
       "https://github.com/decompme/compilers (n64_sn272_0006.tar.gz)", ("cc1n64.exe",), "gcc2.7.2sn0006",
       "PE32 (Windows)", emulator="wibo"),
    _c("sn/cc1n64-2.8.1", "compiler", "SN Systems cc1n64 (GCC 2.8.1)", "2.8.1 (1,0,0,2)", "authentic-n64",
       "implemented-verified", _DECOMPME, "https://github.com/marijnvdwerf/sn64 (release 1,0,0,2)",
       ("cc1n64.exe",), "sn-gcc-2.8", "PE32 (Windows)", emulator="wibo", known_projects="later SN Systems titles"),
    _c("sn/cppn64", "preprocessor", "SN Systems cppn64", "2.7.2", "authentic-n64", "comparative-only", _DECOMPME,
       "https://github.com/decompme/compilers (n64_sn272_0001/0006)", ("cppn64.exe",), "gcc2.7.2sn0006",
       "PE32 (Windows)", emulator="wibo",
       notes="present but not used: the kit's SN lanes preprocess with the host gcc -E (a deviation from the "
             "authentic recipe; tokens agree for ordinary C, line notes and predefined macros may not)"),
    _c("sn/ccn64", "driver", "SN Systems ccn64 driver", "2.7.2", "authentic-n64", "comparative-only", _DECOMPME,
       "https://github.com/decompme/compilers", ("ccn64.exe",), "gcc2.7.2sn0006", "PE32 (Windows)",
       emulator="wibo", notes="the kit calls the stages itself"),
    _c("sn/cc1pln64", "frontend", "SN Systems C++ compiler (cc1pln64)", "2.7.2/2.8.1", "authentic-n64",
       "comparative-only", _DECOMPME, "https://github.com/marijnvdwerf/sn64", ("cc1pln64.exe",), "gcc2.7.2sn0006",
       "PE32 (Windows)", emulator="wibo", language="C++"),
    _c("sn/asn64", "assembler", "SN Systems asn64 assembler", "", "authentic-assembler-linker",
       "implemented-verified", _DECOMPME, "SN kit archives above", ("asn64.exe",), "sn-gcc-2.8", "PE32 (Windows)",
       emulator="wibo", notes="writes SN's own object format (the PsyQ object format shared with the PS1 kits)"),
    _c("psyq-obj-parser", "object-converter", "psyq-obj-parser (SN/PsyQ object to ELF)", "", "reconstructed",
       "implemented-verified", _DECOMPME, "https://github.com/decompme/compilers (psyq-obj-parser.tar.gz)",
       ("psyq-obj-parser",), "sn-gcc-2.8", "x86-64 ELF",
       notes="the same converter serves PS1 PsyQ objects: shared object format, evidence of shared tooling"),
    _c("wibo", "emulator", "wibo (Windows executables on Linux)", "", "reconstructed", "implemented-verified",
       "bundled with the SN kit", "https://github.com/decompals/wibo", ("wibo",), "sn-gcc-2.8", "x86 ELF"),
    _c("sn64-gcc-970404", "compiler", "SN64 GCC 2.7.2 rebuilt from source (970404)", "2.7.2", "reconstructed",
       "implemented-verified", _DECOMPME, "https://github.com/decompals/SN64-gcc (gcc-2.7.2-970404)",
       ("cc1", "cpp", "gcc"), "gcc2.7.2snew", "x86-64 ELF", licence="GPL"),
    _c("modern-asn64", "assembler-macro", "modern-asn64 (asn64 behaviour over GNU as)", "", "reconstructed",
       "implemented-verified", _DECOMPME, "https://github.com/RocketRet/modern-asn64", ("modern-asn64.py",),
       "gcc2.7.2snew", "Python"),
    # ---- other GCC builds
    _c("gcc2.8.1pm", "compiler", "GCC 2.8.1 (Paper Mario build) with its binutils", "2.8.1", "reconstructed",
       "implemented-verified", _DECOMPME, "https://github.com/pmret/gcc-papermario, pmret/binutils-papermario",
       ("gcc", "cc1", "as"), "gcc2.8.1pm", "x86-64 ELF", licence="GPL", known_projects="Paper Mario"),
    _c("egcs-1.1.2-4", "compiler", "EGCS 1.1.2 mips-linux (original binaries)", "2.91.66", "authentic-n64",
       "unavailable", _DECOMPME, "https://github.com/AngheloAlf/egcs_1.1.2-4", ("mips-linux-gcc", "cc1", "as"),
       "egcs_1.1.2-4", "i386 ELF", licence="GPL", notes="needs a 32-bit loader; the rebuilt -4c is driven"),
    _c("egcs-1.1.2-4c", "compiler", "EGCS 1.1.2 rebuilt (gcc 2.91.66, binutils 2.9.5)", "2.91.66", "reconstructed",
       "implemented-verified", _DECOMPME, "https://github.com/decompals/mips-gcc-egcs-2.91.66",
       ("gcc", "cc1", "as"), "egcs_1.1.2-4c", "x86-64 ELF", licence="GPL", known_projects="libultra 2.0L"),
    _c("gcc4.4.0-mips64-elf", "compiler", "GCC 4.4.0 mips64-elf", "4.4.0", "comparative", "comparative-only",
       _DECOMPME, "https://github.com/devwizard64/gcc4.4.0-mips64-elf", ("bin/mips64-elf-gcc",),
       "gcc4.4.0-mips64-elf", "x86-64 ELF", licence="GPL", retail_evidence="none: homebrew"),
    # ---- PlayStation relatives: comparative controls only
    _c("psyq-ps1", "compiler", "PsyQ PS1 SDK compilers (3.3-4.6) and psyq_263_221", "3.3-4.6", "comparative",
       "undistributable-local", _DECOMPME, "https://github.com/mkst/esa (psyq-binaries release)", (), "", "DOS/Windows",
       target="mipsel-psx (little-endian R3000)", licence="Sony/SN Systems SDK; not redistributable",
       retail_evidence="none for N64",
       notes="not installed: proprietary SDK binaries. Relevance to N64: SN Systems built both kits; the N64 asn64 "
             "writes the object format psyq-obj-parser reads for PS1 PsyQ objects"),
    _c("old-gcc-psx", "compiler", "GCC 2.5.7-2.95.2 mipsel / -psx builds", "2.5.7-2.95.2", "comparative",
       "comparative-only", _DECOMPME, "https://github.com/decompals/old-gcc", (), "", "x86-64 ELF",
       target="mipsel (little-endian)", licence="GPL", retail_evidence="none for N64",
       notes="the same GCC versions as KMC/SN N64 compilers for another target: a control for what is GCC "
             "version behaviour and what is the N64 vendors' patches"),
    _c("sdevtc", "compiler", "SN Systems SDevTC (Sega toolchains)", "", "speculation", "speculation",
       "named in the directive", "", (), "", "", target="SH/68k", retail_evidence="none for N64", confidence="low",
       notes="no N64 artefact, project or binary evidence found"),
    # ---- linkers and post-link processing of retail builds
    _c("sdk/makerom", "post-link", "Nintendo SDK makerom / spicy (ROM assembly, segments, boot)", "", "authentic-n64",
       "unavailable", "Nintendo 64 SDK", "", (), "", "IRIX / Windows", licence="Nintendo SDK; not redistributable",
       notes="the kit does not rebuild ROMs; it reads placement from the survey (segments, deltas)"),
    _c("sdk/psylink", "linker", "SN Systems psylink (N64)", "", "authentic-assembler-linker", "unavailable",
       "SN Systems N64 kit", "", (), "", "Windows", licence="SN Systems proprietary",
       notes="not found in the public archives; SN-linked titles are matched per function, not relinked"),
)

BY_ID = {component.ident: component for component in COMPONENTS}


# ---------------------------------------------------------------------------------------------
# Recipes: the lanes the kit drives, stage by stage

@dataclass
class Stage:
    stage: str                     # preprocess, frontend, optimize, codegen, assemble, convert, ...
    component: str                 # Component.ident
    flags: tuple[str, ...] = ()
    input: str = ""
    output: str = ""
    note: str = ""


@dataclass
class Recipe:
    lane: str                      # the pipeline name runs use (ido5.3, kmc+as, custom:..., sn)
    stages: list[Stage]
    env: dict = field(default_factory=dict)
    note: str = ""


def _ido(version: str) -> list[Stage]:
    v = version
    return [Stage("driver", f"ido{v}/cc", ("-c",), "C", "object", "runs every stage below"),
            Stage("preprocess", f"ido{v}/cfe", (), "C", "tokens", "internal preprocessor (acpp with -acpp)"),
            Stage("frontend", f"ido{v}/cfe", (), "tokens", "ucode"),
            Stage("optimize", f"ido{v}/uopt", (), "ucode", "ucode", "skipped at -O0/-g"),
            Stage("codegen", f"ido{v}/ugen", (), "ucode", "binary assembly"),
            Stage("assemble", f"ido{v}/as1", (), "binary assembly", "ELF relocatable", "delay slots, scheduling")]


def _kmc(assembler: list[Stage]) -> list[Stage]:
    return [Stage("driver", "kmc/gcc", ("-S",), "C", "assembly text"),
            Stage("preprocess", "kmc/cpp", (), "C", "tokens"),
            Stage("compile", "kmc/cc1", (), "tokens", "assembly text", "front end, optimiser, code generator"),
            *assembler]


RECIPES: dict[str, Recipe] = {
    "ido5.3": Recipe("ido5.3", _ido("5.3")),
    "ido7.1": Recipe("ido7.1", _ido("7.1")),
    "kmc+as": Recipe("kmc+as", _kmc([Stage("assemble", "kmc/as", (), "assembly text", "ELF relocatable",
                                            "reorder on")])),
    "kmc+builtin": Recipe("kmc+builtin", _kmc([Stage("normalize", "kit/asn64-converter", (), "assembly text",
                                                      "assembly text", "built-in normaliser"),
                                                Stage("assemble", "kmc/as", (), "assembly text", "ELF relocatable")])),
    "kmc+atload": Recipe("kmc+atload", _kmc([Stage("expand", "kit/at-symbol-loads", (), "assembly text",
                                                    "assembly text", "symbol loads through $at (emulated assembler)"),
                                              Stage("assemble", "kmc/as", (), "assembly text", "ELF relocatable",
                                                    "reorder on")])),
    "kmc+asg": Recipe("kmc+asg", _kmc([Stage("assemble", "kmc/as", (), "assembly text", "ELF relocatable",
                                             "-G from the compiler's -G")])),
    "kmc+gas": Recipe("kmc+gas", _kmc([Stage("assemble", "gnu/as-2.9", (), "assembly text", "ELF relocatable",
                                             "-G from the compiler's -G")])),
    "kmc+atmacro": Recipe("kmc+atmacro", _kmc([Stage("expand", "kit/at-symbol-loads", (), "assembly text",
                                                     "assembly text", "symbol loads through $at (emulated assembler)"),
                                               Stage("expand", "kit/at-branch-macros", (), "assembly text",
                                                     "assembly text", "branch comparisons through $at"),
                                               Stage("assemble", "kmc/as", (), "assembly text", "ELF relocatable",
                                                     "reorder on")])),
    "kmc+asn64": Recipe("kmc+asn64", _kmc([Stage("convert", "kit/asn64-converter", (), "assembly text",
                                                  "assembly text"),
                                            Stage("assemble", "binutils/as", ("--no-pad-sections",), "assembly text",
                                                  "ELF relocatable")])),
    "sn": Recipe("sn", [Stage("preprocess", "host/gcc-E", ("-E", "-P", "-undef"), "C", "tokens",
                              "host preprocessor, not SN cppn64"),
                        Stage("compile", "sn/cc1n64-2.8.1", (), "tokens", "assembly text"),
                        Stage("assemble", "sn/asn64", (), "assembly text", "SN object"),
                        Stage("convert", "psyq-obj-parser", ("-b", "-n"), "SN object", "ELF relocatable")]),
    "custom:gcc272sn0001": Recipe("custom:gcc272sn0001", [
        Stage("preprocess", "host/gcc-E", (), "C", "tokens"), Stage("compile", "sn/cc1n64-2.7.2-0001"),
        Stage("assemble", "sn/asn64"), Stage("convert", "psyq-obj-parser", ("-b", "-n", "-s"))]),
    "custom:gcc272sn0004": Recipe("custom:gcc272sn0004", [
        Stage("preprocess", "host/gcc-E", (), "C", "tokens"), Stage("compile", "sn/cc1n64-2.7.2-0004"),
        Stage("assemble", "sn/asn64", ("-G0",)), Stage("convert", "psyq-obj-parser", ("-b", "-n"))]),
    "custom:gcc272sn0006": Recipe("custom:gcc272sn0006", [
        Stage("preprocess", "host/gcc-E", (), "C", "tokens"), Stage("compile", "sn/cc1n64-2.7.2-0006"),
        Stage("assemble", "sn/asn64", ("-G0",)), Stage("convert", "psyq-obj-parser", ("-b", "-n", "-s"))]),
    "custom:gcc272snew": Recipe("custom:gcc272snew", [
        Stage("preprocess", "sn64-gcc-970404"), Stage("compile", "sn64-gcc-970404"),
        Stage("convert", "modern-asn64"), Stage("assemble", "binutils/as")]),
    "custom:gcc281pm": Recipe("custom:gcc281pm", [Stage("driver", "gcc2.8.1pm", ("-c",), "C", "ELF relocatable")]),
    "custom:egcs112": Recipe("custom:egcs112", [Stage("driver", "egcs-1.1.2-4c", ("-c",), "C", "ELF relocatable")]),
    "custom:ido52": Recipe("custom:ido52", [Stage("driver", "ido5.2/driver", ("-c",)),
                                            Stage("emulate", "qemu-irix")]),
    "custom:ido60": Recipe("custom:ido60", [Stage("driver", "ido6.0/driver", ("-c",)),
                                            Stage("emulate", "qemu-irix")]),
    "custom:mipspro744": Recipe("custom:mipspro744", [Stage("driver", "mips_pro_744/driver", ("-c",)),
                                                      Stage("emulate", "qemu-irix")]),
}

HOST_COMPONENTS = {"host/gcc-E": "the host system's gcc -E (a modern GNU cpp)"}


# ---------------------------------------------------------------------------------------------
# Inspection against a toolchain root

def _format(path: Path) -> str:
    try:
        head = path.read_bytes()[:64]
    except OSError:
        return "unreadable"
    if head[:4] == b"\x7fELF":
        bits = {1: "32-bit", 2: "64-bit"}.get(head[4], "?")
        endian = {1: "little-endian", 2: "big-endian"}.get(head[5], "?")
        machine = int.from_bytes(head[18:20], "little" if head[5] == 1 else "big")
        arch = {3: "i386", 8: "MIPS", 62: "x86-64"}.get(machine, f"machine {machine}")
        return f"ELF {bits} {endian} {arch}"
    if head[:2] == b"MZ":
        return "PE (Windows)"
    if head[:2] in (b"\x01\x60", b"\x01\x62", b"\x60\x01"):
        return "MIPS ECOFF"
    if head[:2] == b"#!":
        return "script"
    return "data"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(root: Path | None, digests: bool = True) -> list[dict]:
    """Every component with what a toolchain root holds of it: files found, their formats and
    SHA-256 digests."""
    out = []
    for component in COMPONENTS:
        row = asdict(component)
        found = []
        if root is not None and component.directory:
            base = Path(root) / component.directory
            for name in component.files:
                path = base / name
                if path.is_file():
                    found.append({"file": name, "format": _format(path), "bytes": path.stat().st_size,
                                  **({"sha256": _digest(path)} if digests else {})})
        kit_internal = not component.files and component.origin.startswith("n64dk/")
        row["installed"] = kit_internal or (bool(component.files) and len(found) == len(component.files))
        row["found"] = found
        out.append(row)
    return out


def recipe_identity(recipe: Recipe, root: Path | None, flags: list[str] | None = None,
                    inspected: list[dict] | None = None) -> dict:
    """The recipe with every stage's component digests, and one SHA-256 over all of it (stage
    order, components, file digests, flags, environment).  Two recipes with the same identity
    build the same way; two labels that share all but one component differ by that stage."""
    rows = {row["ident"]: row for row in (inspected if inspected is not None else inspect(root))}
    stages = []
    for stage in recipe.stages:
        row = rows.get(stage.component)
        files = [(item["file"], item.get("sha256")) for item in (row or {}).get("found", [])]
        stages.append({"stage": stage.stage, "component": stage.component, "flags": list(stage.flags),
                       "input": stage.input, "output": stage.output,
                       "provenance": (row or {}).get("provenance", "host" if stage.component in HOST_COMPONENTS else "?"),
                       "files": files, "installed": bool(row and row["installed"]) or stage.component in HOST_COMPONENTS})
    material = {"lane": recipe.lane, "stages": stages, "env": recipe.env, "flags": list(flags or [])}
    identity = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    return {"lane": recipe.lane, "identity": identity, "stages": stages, "flags": list(flags or []),
            "runnable": all(stage["installed"] for stage in stages)}


def shared_stages(a: str, b: str) -> dict:
    """Which stages two recipes share and which differ: a difference in output between them is
    attributable to the differing stages only."""
    first, second = RECIPES[a], RECIPES[b]
    left = {stage.stage: stage.component for stage in first.stages}
    right = {stage.stage: stage.component for stage in second.stages}
    names = sorted(set(left) | set(right))
    return {"same": [name for name in names if left.get(name) == right.get(name)],
            "different": [{"stage": name, a: left.get(name), b: right.get(name)} for name in names
                          if left.get(name) != right.get(name)]}


def catalogue_json(root: Path | None, digests: bool = True) -> dict:
    rows = inspect(root, digests)                     # once: every recipe reads the same inspection
    with_digests = rows if digests else inspect(root)
    return {"provenance_classes": PROVENANCE, "dispositions": DISPOSITIONS, "roles": ROLES,
            "components": rows,
            "recipes": {lane: recipe_identity(recipe, root, inspected=with_digests) for lane, recipe in RECIPES.items()},
            "host_components": HOST_COMPONENTS}
