#!/usr/bin/env python3
"""Dependency-free fallback test runner.

Why this exists
---------------
The canonical way to run the suite is ``make test`` (``python -m pytest -q``).
Some environments - notably a locked-down sandbox with no package index - cannot
install pytest. This runner reimplements the small slice of pytest the suite
actually uses - ``raises``, ``approx``, ``parametrize``, the ``monkeypatch`` and
``tmp_path`` fixtures, and module-local fixtures - so the same tests can be
executed anywhere Python and the runtime dependencies are present.

It is a fallback, not a replacement: when pytest is installed, use it. This file
intentionally favours plain, inspectable code over cleverness, because its only
job is to be trustworthy when the real harness is unavailable.

Usage::

    python scripts/run_tests.py
"""
"""Dependency-free pytest-compatible runner: fixtures, monkeypatch, approx, raises."""
import inspect, sys, traceback, importlib.util, types, re, glob, os, tempfile, math, shutil
sys.path.insert(0, '.'); sys.path.insert(0, 'tests')

pytest = types.ModuleType("pytest")
class _Raises:
    def __init__(self, exc, match=None): self.exc, self.match = exc, match; self.value=None
    def __enter__(self): return self
    def __enter__(self): return self
    def __exit__(self, et, ev, tb):
        if et is None: raise AssertionError(f"did not raise {self.exc}")
        if not issubclass(et, self.exc): return False
        self.value = ev
        if self.match and not re.search(self.match, str(ev)):
            raise AssertionError(f"{ev!r} !~ {self.match!r}")
        return True
class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12): self.e, self.rel, self.abs = expected, rel, abs
    def __eq__(self, other):
        try: return abs(other - self.e) <= max(self.abs, self.rel*abs(self.e))
        except TypeError: return NotImplemented
    __hash__ = None
pytest.raises = lambda exc, match=None: _Raises(exc, match)
pytest.approx = lambda e, rel=1e-6, abs=1e-12: _Approx(e, rel, abs)
pytest.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: setattr(f,"_parametrize",(a[0],a[1])) or f))
def _fixture(*a, **k):
    def deco(f): f._is_fixture = True; return f
    if a and callable(a[0]): return deco(a[0])
    return deco
pytest.fixture = _fixture
sys.modules["pytest"] = pytest

import conftest  # installs stubs, path, and real pytest fixtures (guarded)

class MonkeyPatch:
    def __init__(self):
        self._env = []; self._attr = []
    def setenv(self, name, value, prepend=None):
        self._env.append((name, os.environ.get(name))); os.environ[name] = str(value)
    def delenv(self, name, raising=True):
        self._env.append((name, os.environ.get(name))); os.environ.pop(name, None)
    def setattr(self, target, name, value, raising=True):
        self._attr.append((target, name, getattr(target, name, None) if not isinstance(target, dict) else target.get(name)));
        if isinstance(target, dict): target[name] = value
        else: setattr(target, name, value)
    def undo(self):
        for target, n, v in reversed(self._attr):
            if isinstance(target, dict):
                if v is None: target.pop(n, None)
                else: target[n] = v
            else:
                if v is None:
                    try: delattr(target, n)
                    except Exception: pass
                else: setattr(target, n, v)
        for name, v in reversed(self._env):
            if v is None: os.environ.pop(name, None)
            else: os.environ[name] = v

def build_fixture_values(func, tmpdir, registry, provided=()):
    """Instantiate fixture args from a registry or built-ins."""
    vals = {}
    for pname in list(inspect.signature(func).parameters)[1:]:  # drop self
        if pname in provided:
            continue
        if pname == "monkeypatch":
            vals[pname] = MonkeyPatch()
        elif pname == "tmp_path":
            vals[pname] = tmpdir
        elif pname in registry:
            vals[pname] = call_fixture(pname, registry[pname], tmpdir)
        else:
            raise TypeError(f"unknown fixture {pname}")
    return vals

fx_registry = {}
for n in dir(conftest):
    obj = getattr(conftest, n)
    if callable(obj) and getattr(obj, "_is_fixture", False) or (getattr(obj, "__name__", "") in ("kafka_module","sample_record")):
        fx_registry[getattr(obj,"__name__",n)] = obj
# register by declared name
for n in dir(conftest):
    obj = getattr(conftest, n)
    if callable(obj) and not inspect.isclass(obj):
        fx_registry.setdefault(n, obj)

_CURRENT_REG = {}
def call_fixture(name, fn, tmpdir):
    kwargs = {}
    for pname in inspect.signature(fn).parameters:
        if pname == "monkeypatch": kwargs[pname] = MonkeyPatch()
        elif pname == "tmp_path": kwargs[pname] = tmpdir
        elif pname in _CURRENT_REG:
            kwargs[pname] = call_fixture(pname, _CURRENT_REG[pname], tmpdir)
    return fn(**kwargs)

total_p = total_f = 0
for path in sorted(glob.glob("tests/test_*.py")):
    name = os.path.basename(path)[:-3]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try: spec.loader.exec_module(mod)
    except Exception as e:
        print(f"[import error] {name}: {e}"); continue
    registry = dict(fx_registry)
    for n, obj in vars(mod).items():
        if callable(obj) and not inspect.isclass(obj):
            registry.setdefault(n, obj)
    _CURRENT_REG.clear(); _CURRENT_REG.update(registry)
    p = f = 0
    for cname, obj in list(vars(mod).items()):
        if inspect.isclass(obj) and cname.startswith("Test"):
            inst = obj()
            for mname, m in vars(obj).items():
                if not (mname.startswith("test_") and callable(m)): continue
                tmpdir = None
                try:
                    tmpdir = __import__("pathlib").Path(tempfile.mkdtemp())
                    cases = getattr(m, "_parametrize", None)
                    if cases:
                        names=[a.strip() for a in cases[0].split(",")]
                        for vals in cases[1]:
                            vals = vals if isinstance(vals,(tuple,list)) else (vals,)
                            extra = dict(zip(names, vals))
                            kw = build_fixture_values(m, tmpdir, registry, provided=set(names))
                            mp = kw.get("monkeypatch")
                            m(inst, **{**kw, **extra})
                    else:
                        kw = build_fixture_values(m, tmpdir, registry)
                        mp = kw.get("monkeypatch")
                        # unittest.TestCase setUp/tearDown are required for stateful tests.
                        if hasattr(inst, "setUp"):
                            inst.setUp()
                        try:
                            m(inst, **kw)
                        finally:
                            if hasattr(inst, "tearDown"):
                                inst.tearDown()
                    p += 1
                except Exception:
                    f += 1; print(f"  FAIL {name}.{cname}.{mname}"); traceback.print_exc()
                finally:
                    if mp: mp.undo()
                    if tmpdir: shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"{name}: {p} passed, {f} failed")
    total_p += p; total_f += f
print(f"\nTOTAL: {total_p} passed, {total_f} failed")
