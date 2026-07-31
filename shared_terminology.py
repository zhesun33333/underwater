"""
Shared L1/L2/L3 terminology for underwater acoustic classification.

This is the SINGLE SOURCE OF TRUTH for:
  - L3_PRECISE:  pipeline QA generation terms (archive/pipeline_step2_qa.py,
                 archive/pipeline_ship_step2_qa.py)
  - L3_SHOULD:   scorer evaluation terms (testsite/core/scorer.py)
  - L1/L2 descriptive terms for T3 reasoning generation

When terminology changes, update ONLY this file — all consumers re-import.

Two-tier design:
  - L3_PRECISE:  one most-accurate academic term per concept (used in SFT data)
  - L3_SHOULD:   expanded synonym set (used in scorer for generous matching)
"""

# ============================================================
# L1 descriptive terms — participle/adjective form for T3 templates
# ============================================================
L1_ACTIVE_TERMS  = ["actively transmitted", "deliberately emitted", "artificially generated"]
L1_PASSIVE_TERMS = ["passively received", "target-radiated", "naturally emitted"]

# ============================================================
# L2 descriptive terms — noun phrases keyed by L2 label
# ============================================================
L2_ACTIVE_TERMS = {
    "detection pulse":      ["detection pulse characteristics", "pulse detection mode",
                             "sonar pulse properties"],
    "communication signal": ["underwater acoustic communication features",
                             "digital communication modulation",
                             "communication modulation scheme"],
}
L2_PASSIVE_TERMS = {
    "ship-radiated noise": ["ship-radiated noise characteristics",
                            "vessel-radiated noise properties",
                            "target-radiated noise pattern"],
}

# ============================================================
# L3 PRECISE — for QA generation (SFT training data)
# One most-accurate term per acoustic concept; randomly sampled 2-3 per sample
# ============================================================
L3_PRECISE = {
    # ── Detection pulse ──
    "CW":  ["single-frequency", "continuous wave", "constant pitch",
            "single tone", "narrowband"],
    "LFM": ["linear frequency modulation", "linear chirp", "linear sweep",
            "frequency glide", "bandwidth broadening", "upsweep", "downsweep"],
    "HFM": ["hyperbolic frequency modulation", "hyperbolic chirp", "hyperbolic sweep",
            "nonlinear sweep", "decelerating sweep rate", "fast-to-slow sweep",
            "Doppler-invariant"],
    # ── Communication ──
    "2FSK": ["dual-tone", "binary modulation", "frequency shift keying",
             "frequency shift", "tone alternation"],
    "4FSK": ["four tones", "quaternary modulation", "frequency shift keying",
             "frequency shift", "multi-tone alternation"],
    "BPSK": ["phase reversal", "phase shift keying", "binary", "two phase states",
             "abrupt phase transition", "constant amplitude"],
    "QPSK": ["phase shift keying", "four phase states", "quaternary",
             "multi-state phase transition", "constant amplitude"],
    "OFDM": ["subcarriers", "multi-carrier", "orthogonal", "frequency division multiplexing",
             "parallel transmission", "dense subcarriers"],
    # ── Ship-radiated noise ──
    "cargo":   ["clear harmonic structure", "shaft rate", "shaft rate periodicity",
                "low-speed large-displacement", "large merchant vessel",
                "regular rhythmic pattern", "harmonic families"],
    "cruise":  ["dense machinery noise", "multiple machinery sources", "high-speed",
                "multiple engine units", "broadband noise", "mid-to-high frequency band",
                "passenger vessel signature"],
    "fishing": ["small-displacement", "simple machinery configuration", "sparse tonals",
                "diesel engine", "low source level", "low-order harmonics",
                "limited tonal content"],
    "warship": ["high-power", "high source level", "naval vessel", "multi-shaft propulsion",
                "multiple harmonic families", "dense and complex tonals",
                "energy concentrated at low frequencies"],
    "underwater_target": ["underwater vehicle", "smooth broadband spectrum",
                          "low source level", "predominantly broadband",
                          "few tonal components", "aperiodic"],
}

# ============================================================
# L3 SHOULD — for scorer reasoning-quality evaluation
# Superset of L3_PRECISE: includes all valid English variants a model
# might reasonably output for each class
# ============================================================
L3_SHOULD = {
    # ── Detection pulse ──
    "CW":  {"single-frequency", "continuous wave", "constant pitch",
            "single tone", "narrowband", "steady frequency",
            "unchanging pitch", "unmodulated", "pure tone",
            "fixed frequency", "constant frequency"},
    "LFM": {"linear frequency modulation", "linear chirp", "linear sweep",
            "frequency glide", "bandwidth broadening", "upsweep",
            "downsweep", "sweeping frequency", "rising pitch",
            "falling pitch", "frequency ramp", "chirp"},
    "HFM": {"hyperbolic frequency modulation", "hyperbolic chirp",
            "hyperbolic sweep", "nonlinear sweep", "decelerating sweep rate",
            "fast-to-slow sweep", "Doppler-invariant", "Doppler invariant",
            "nonlinear chirp", "varying sweep rate"},
    # ── Communication ──
    "2FSK": {"dual-tone", "binary modulation", "frequency shift keying",
             "frequency shift", "tone alternation", "two frequencies",
             "dual frequency", "alternating frequency", "binary shift",
             "frequency switching"},
    "4FSK": {"four tones", "quaternary modulation", "frequency shift keying",
             "frequency shift", "multi-tone alternation", "four frequencies",
             "alternating frequencies", "multi-level FSK",
             "frequency switching"},
    "BPSK": {"phase reversal", "phase shift keying", "binary",
             "two phase states", "abrupt phase transition",
             "constant amplitude", "phase flip", "phase inversion",
             "phase discontinuity", "binary phase", "two-phase",
             "steady amplitude"},
    "QPSK": {"phase shift keying", "four phase states", "quaternary",
             "multi-state phase transition", "constant amplitude",
             "four-phase", "quadrature modulation", "steady amplitude"},
    "OFDM": {"subcarriers", "multi-carrier", "orthogonal",
             "frequency division multiplexing", "parallel transmission",
             "dense subcarriers", "orthogonal carriers",
             "multiple subcarriers", "simultaneous transmission",
             "dense carriers"},
    # ── Ship-radiated noise ──
    "cargo":   {"clear harmonic structure", "shaft rate",
                "shaft rate periodicity", "low-speed large-displacement",
                "large merchant vessel", "regular rhythmic pattern",
                "harmonic families", "shaft frequency", "shaft harmonics",
                "propeller rate", "blade rate", "slow speed",
                "large vessel", "rich harmonics", "harmonic lines",
                "merchant", "tonal"},
    "cruise":  {"dense machinery noise", "multiple machinery sources",
                "high-speed", "multiple engine units", "broadband noise",
                "mid-to-high frequency band", "passenger vessel signature",
                "passenger", "high frequency", "machinery",
                "multiple engines", "dense tonals", "mechanical noise"},
    "fishing": {"small-displacement", "simple machinery configuration",
                "sparse tonals", "diesel engine", "low source level",
                "low-order harmonics", "limited tonal content",
                "small vessel", "small boat", "diesel",
                "low power", "weak signal", "low amplitude",
                "low intensity", "simple harmonics", "few harmonics",
                "trawler"},
    "warship": {"high-power", "high source level", "naval vessel",
                "multi-shaft propulsion", "multiple harmonic families",
                "dense and complex tonals",
                "energy concentrated at low frequencies",
                "naval", "military", "warship", "high intensity",
                "strong signal", "multiple shafts", "combat",
                "low frequency emphasis", "powerful"},
    "underwater_target": {"underwater vehicle", "smooth broadband spectrum",
                          "low source level", "predominantly broadband",
                          "few tonal components", "aperiodic",
                          "submerged", "UUV", "AUV",
                          "low amplitude", "low intensity",
                          "featureless", "lacking harmonics",
                          "structureless", "quiet"},
}

# Auto-derived: global set of all L3 distinguishing terms
_ALL_L3_TERMS = set()
for _s in L3_SHOULD.values():
    _ALL_L3_TERMS.update(_s)


def get_should_not(l3: str) -> set:
    """Return terms that should NOT appear in reasoning for a given L3 class."""
    own = L3_SHOULD.get(l3, set())
    return _ALL_L3_TERMS - own
