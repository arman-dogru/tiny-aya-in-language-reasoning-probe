from probe.metrics import lid_clean, parse_answer, repetition_score, split_trace
from probe.stats import mcnemar_exact, paired_bootstrap_diff, wilson


def test_split_and_parse_math():
    tr, ans = split_trace("Step 1: 16-7=9. 9*2=18.\n\n#### 18", think_tags=False)
    assert tr.endswith("18.") and parse_answer(ans, "math") == "18"
    assert parse_answer("#### $1,250.00", "math") == "1250"
    assert parse_answer("#### 2.5", "math") == "2.5"
    assert parse_answer("no answer line 42", "math") is None


def test_parse_letter_and_think_tags():
    assert parse_answer("#### **B**", "piqa") == "B"
    assert parse_answer("#### (c)", "piqa") == "C"
    assert parse_answer("#### E", "piqa") is None
    tr, ans = split_trace("hmm ok\n</think>\n\n#### D", think_tags=True)
    assert tr == "hmm ok" and parse_answer(ans, "piqa") == "D"
    tr, ans = split_trace("loop loop", think_tags=True)
    assert ans == ""


def test_last_answer_line_wins():
    tr, ans = split_trace("#### 3 is wrong, recompute.\n#### 4", think_tags=False)
    assert parse_answer(ans, "math") == "4"


def test_repetition():
    assert repetition_score("a b c") is None
    assert repetition_score("one two three four five six seven eight nine ten") == 0.0
    loop = " ".join(["so the answer is nine"] * 20)
    assert repetition_score(loop) > 0.7


def test_lid_clean_strips_math():
    assert lid_clean(r"Donc \[ 3 + 5 = 8 \] et 16 - 7 = 9 #### 18") == "Donc et"


def test_stats():
    lo, hi = wilson(10, 20)
    assert 0.29 < lo < 0.30 and 0.70 < hi < 0.71
    assert wilson(0, 0)[0] != wilson(0, 0)[0]  # nan
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 6) - 0.03125) < 1e-12
    m, lo, hi = paired_bootstrap_diff([1, 1, 1, 0], [0, 0, 1, 0], iters=2000)
    assert m == 0.5 and lo <= m <= hi


def test_aya_thinker_tags():
    tr, ans = split_trace("Je réfléchis.<|END_THINKING|><|START_RESPONSE|>#### 7", think_tags=True,
                          think_close="<|END_THINKING|>")
    assert tr == "Je réfléchis." and parse_answer(ans, "math") == "7"


def test_prefill_and_messages():
    from probe.run import prefill, user_message
    it = dict(question="Q?", task="math", lang="yo", gold="1")
    assert prefill(it, "l2", False) == "Ìdáhùn lẹ́sẹsẹ:"
    assert prefill(it, "en", False) == "Step-by-Step Answer:"
    assert prefill(it, "l2", True, "", "<|END_THINKING|>") == "Ìdáhùn lẹ́sẹsẹ:"
    assert prefill(it, "direct", True) == "<think>\n\n</think>\n\n#### "
    assert "same language as the question" in user_message(it, "l2")
    assert "in English" in user_message(it, "en")


def test_lenient_parser():
    from probe.metrics import parse_answer_lenient as pl
    assert pl("Réponse finale :\nB. La quantité d’air diminue.", "piqa") == "B"
    assert pl("A. foo is wrong. The right one is **C**.", "piqa") == "C"
    assert pl("so 9 x 2 = 18.", "math") == "18"
    assert pl("#### 7\nlater 9", "math") == "7"  # strict answer line wins
    assert pl("I think so", "piqa") is None
