"""Hand-computed fixtures for the code that produces the headline numbers
(score_run -> paired -> add_multiplicity, summarize). Each test is written to kill a
specific mutation from review round 1 (m1)."""
import json
from collections import Counter

import pytest

from probe.score import add_multiplicity, paired, score_run, src_of, summarize
from probe.stats import paired_bootstrap_diff, signflip_cluster_p


class FakeLID:
    """Labels a trace by a marker word so tests control the language ID exactly."""
    MAP = {"ENGLISH": "eng_Latn", "KISWAHILI": "swh_Latn", "YORUBA": "yor_Latn", "FRANCAIS": "fra_Latn"}

    def label(self, text, min_words=3):
        for k, v in self.MAP.items():
            if k in text:
                return v, 0.99
        return "und", 0.0

    def segment_shares(self, text):
        lab, _ = self.label(text)
        return Counter({lab: 1.0}) if lab != "und" else Counter()


def rec(item_id, lang, arm, completion, gold="5", task="math"):
    return dict(id=item_id, task=task, lang=lang, gold=gold, model="m", arm=arm, completion=completion,
                gen_tokens=10, cap_hit=False, budget_forced=False, wall_s=1.0)


def write(tmp_path, recs):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return str(p)


def test_src_of_groups_translations():
    # mutation "cluster key = full id" must change these
    assert src_of("mgsm-012-yo", "yo") == "mgsm-012"
    assert src_of("piqa-000003-fr-CA", "fr-CA") == "piqa-000003"
    assert src_of("piqa-000003-fr", "fr") == "piqa-000003"


def test_l2_rate_does_not_count_english(tmp_path):
    # mutation "L2 rate counts eng_Latn" must fail here
    rows = score_run(write(tmp_path, [
        rec("mgsm-001-sw", "sw", "l2", "ENGLISH words words words\n#### 5"),
        rec("mgsm-001-yo", "yo", "l2", "YORUBA words words words\n#### 5"),
        rec("mgsm-001-en", "en", "l2", "ENGLISH words words words\n#### 5"),
    ]), "", FakeLID())
    by = {r["id"]: r for r in rows}
    assert by["mgsm-001-sw"]["l2"] == 0 and by["mgsm-001-sw"]["en_trace"] == 1
    assert by["mgsm-001-yo"]["l2"] == 1
    assert by["mgsm-001-en"]["l2"] == 1  # English prompt reasoned in English counts as L2
    assert all(r["src"] == "mgsm-001" for r in rows)


def fixture_rows(tmp_path):
    """3 source items x {sw, yo} x {l2, en}. Hand-computed:
    item 1: l2 wrong/wrong, en right/right  -> cluster diff -2
    item 2: l2 right/wrong, en right/right  -> cluster diff -1
    item 3: l2 right/right, en wrong/right  -> cluster diff +1
    l2 = 3/6, en = 5/6, discordant: l2-only = 1 (003-sw), en-only = 3 (001-sw, 001-yo, 002-yo)."""
    plan = {("001", "sw"): (0, 1), ("001", "yo"): (0, 1), ("002", "sw"): (1, 1), ("002", "yo"): (0, 1),
            ("003", "sw"): (1, 0), ("003", "yo"): (1, 1)}
    recs = []
    for (i, l), (ok_l2, ok_en) in plan.items():
        tag = {"sw": "KISWAHILI", "yo": "YORUBA"}[l]
        recs.append(rec(f"mgsm-{i}-{l}", l, "l2", f"{tag} a b c\n#### {5 if ok_l2 else 4}"))
        recs.append(rec(f"mgsm-{i}-{l}", l, "en", f"ENGLISH a b c\n#### {5 if ok_en else 4}"))
    return score_run(write(tmp_path, recs), "", FakeLID())


def test_paired_counts_and_direction(tmp_path):
    rows = fixture_rows(tmp_path)
    pr = paired(rows, "l2", "en")
    v = pr["m|l2-en|math|group:low"]
    assert v["n"] == 6 and v["n_clusters"] == 3
    assert v["acc_a"] == pytest.approx(3 / 6) and v["acc_b"] == pytest.approx(5 / 6)
    # mutation "paired() counts `y and x` as discordant" must fail here
    assert (v["a_only"], v["b_only"]) == (1, 3)
    assert sorted(v["b_only_ids"]) == ["mgsm-001-sw", "mgsm-001-yo", "mgsm-002-yo"]
    assert v["a_only_ids"] == ["mgsm-003-sw"]
    assert v["diff"] == pytest.approx(-2 / 6)
    # cluster sign-flip over sums (-2, -1, +1): |T|=2; 8 sign vectors, |T*|>=2 for 6 -> 0.75
    assert v["p_cluster"] == pytest.approx(0.75)


def test_clustering_changes_the_interval():
    # Perfectly correlated translations: clustered resampling must give a WIDER interval
    # than treating the 30 rows as independent (mutation "clustering ignored" must fail).
    a = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0] * 3
    b = [0] * 30
    clusters = list(range(10)) * 3
    _, lo_c, hi_c = paired_bootstrap_diff(a, b, iters=4000, clusters=clusters)
    _, lo_r, hi_r = paired_bootstrap_diff(a, b, iters=4000)
    assert (hi_c - lo_c) > 1.3 * (hi_r - lo_r)
    # and the cluster test is far less confident than a row-level one would be
    assert signflip_cluster_p(a, b, clusters) == pytest.approx(2 / 2 ** 5)


def test_multiplicity_skips_duplicate_rows(tmp_path):
    rows = fixture_rows(tmp_path)
    pr = add_multiplicity(paired(rows, "l2", "en"))
    fam = pr["_family_l2-en"]
    # cells: math|sw, math|yo, math|group:low, all|group:low (same items as math|group:low)
    assert fam["n_tests"] == 3
    assert pr["m|l2-en|all|group:low"]["duplicate_of"] == "m|l2-en|math|group:low"
    for k in fam["tests"]:
        assert pr[k]["p_cluster_holm"] >= pr[k]["p_cluster"]


def test_summarize_is_order_independent(tmp_path):
    rows = fixture_rows(tmp_path)
    s1 = json.dumps(summarize(rows), sort_keys=True)
    s2 = json.dumps(summarize(list(reversed(rows))), sort_keys=True)
    assert s1 == s2
    c = summarize(rows)["m|l2|math|group:low"]
    assert c["n"] == 6 and c["acc"] == pytest.approx(0.5) and c["l2"] == pytest.approx(1.0)


def test_empty_trace_file_gives_clear_error(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit, match="contains no traces"):
        score_run(str(p), "", FakeLID())


def test_avg_rank_and_spearman_ties():
    from probe.figures import spearman
    from probe.stats import avg_rank
    assert list(avg_rank([10, 20, 20, 5])) == [1.0, 2.5, 2.5, 0.0]
    # with ties, ordinal ranks (argsort of argsort) give 1.0 here; tie-averaged gives 0.866
    assert spearman([0, 0, 1], [1, 2, 3]) == pytest.approx(0.8660254, abs=1e-6)
