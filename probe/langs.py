"""Language metadata for the probe.

`opener` is the in-language "step-by-step answer" phrase used to prefill the
reasoning in the L2-forced arm. Sources:
- en / fr / sw: MGSM train split exemplars (juletxara/mgsm), human-translated.
- yo: AfriMGSM dev exemplar (masakhane/afrimgsm), human-translated.
- zu: AfriMGSM's Zulu exemplar has no such phrase, so this one was written by
  the project author ("Impendulo" is taken from the AfriMGSM exemplar,
  "ngezinyathelo" = "in steps"). It has NOT been checked by a native speaker.
"""

LANGS = {
    # key: (display name, GlotLID label(s) counted as "in target language", resource group, opener)
    "en": dict(name="English", lid=["eng_Latn"], group="high (control)", opener="Step-by-Step Answer:"),
    "fr": dict(name="French (France)", lid=["fra_Latn"], group="high", opener="Réponse étape par étape :"),
    "fr-CA": dict(name="French (Canada)", lid=["fra_Latn"], group="high", opener="Réponse étape par étape :"),
    "sw": dict(name="Swahili", lid=["swh_Latn", "swa_Latn", "swc_Latn"], group="low", opener="Jibu la Hatua kwa Hatua:"),
    "yo": dict(name="Yoruba", lid=["yor_Latn"], group="low", opener="Ìdáhùn lẹ́sẹsẹ:"),
    "zu": dict(name="Zulu", lid=["zul_Latn"], group="low", opener="Impendulo ngezinyathelo:"),
}

# Nguni languages GlotLID confuses with Zulu; used only for the *lenient* L2 rate.
LENIENT_EXTRA = {"zu": ["xho_Latn", "nbl_Latn", "ssw_Latn", "nde_Latn"]}

EN_OPENER = LANGS["en"]["opener"]
