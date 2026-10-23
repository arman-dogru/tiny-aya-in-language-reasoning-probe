# Re-applies the 6 single-line mutations from review round 1 (m1) to a temp copy and runs the tests.
# Usage: python scripts/mutation_check.py  (expects every line to say "killed")
import subprocess, shutil, sys, os
muts = [
 ("probe/stats.py", "    if clusters is None:\n", "    if True:\n", "clustering ignored"),
 ("probe/score.py", "cc = sum(1 for x, y in zip(ca, cb) if y and not x)", "cc = sum(1 for x, y in zip(ca, cb) if y and x)", "paired counts y and x"),
 ("probe/score.py", "src=src_of(r[\"id\"], r[\"lang\"]),", "src=r[\"id\"],", "cluster key = full id"),
 ("probe/score.py", 'l2=int(lab in tgt) if', 'l2=int(lab in tgt + ["eng_Latn"]) if', "L2 rate counts eng_Latn"),
 ("probe/stats.py", "    return min(1.0, 2 * p)", "    return min(1.0, p)", "mcnemar loses x2"),
 ("probe/metrics.py", "        v = n[-1].replace", "        v = n[0].replace", "lenient math takes first number"),
]
res=[]
for f, a, b, name in muts:
    d="/tmp/mutrepo"; shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(".", d, ignore=shutil.ignore_patterns("tools", ".git", "results", "docs"))
    s=open(f"{d}/{f}").read(); assert a in s, name; open(f"{d}/{f}","w").write(s.replace(a,b,1))
    r=subprocess.run([sys.executable,"-m","pytest","-q","-x","tests"],cwd=d,capture_output=True,text=True)
    res.append((name, "killed" if r.returncode else "SURVIVED"))
for n,r in res: print(f"{r:9s} {n}")
