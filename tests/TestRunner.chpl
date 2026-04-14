/*
 * TestRunner.chpl - Minimal TAP test framework (no external dependencies)
 *
 * TAP (Test Anything Protocol) output - compatible with most CI systems.
 */
module TestRunner {
  use Math;

  var _pass: int = 0;
  var _fail: int = 0;
  var _total: int = 0;
  var _suite: string = "";

  proc beginSuite(name: string) {
    _suite = name;
    writeln("# Suite: ", name);
  }

  proc endSuite() { writeln(); }

  proc printSummary() {
    writeln("# ");
    writeln("# Total: ", _total, "  Passed: ", _pass, "  Failed: ", _fail);
    if _fail == 0 then {
      writeln("# RESULT: ALL TESTS PASS ");
    } else {
      writeln("# RESULT: ", _fail, " TEST(S) FAILED ");
    }
  }

  proc allPassed(): bool { return _fail == 0; }

  proc assertTrue(cond: bool, msg: string = "") {
    _total += 1;
    if cond { _pass += 1; writeln("ok ", _total, " - ", msg); }
    else     { _fail += 1; writeln("not ok ", _total, " - ", msg);
                writeln("#   FAIL: condition is false"); }
  }

  proc assertFalse(cond: bool, msg: string = "") {
    assertTrue(!cond, msg);
  }

  proc assertEqual(a: real, b: real, tol: real = 1e-9, msg: string = "") {
    _total += 1;
    if abs(a - b) <= tol {
      _pass += 1; writeln("ok ", _total, " - ", msg);
    } else {
      _fail += 1; writeln("not ok ", _total, " - ", msg);
      writeln("#   FAIL: ", a, " != ", b, "  (tol=", tol, ")");
    }
  }

  proc assertEqual(a: int, b: int, msg: string = "") {
    _total += 1;
    if a == b { _pass += 1; writeln("ok ", _total, " - ", msg); }
    else       { _fail += 1; writeln("not ok ", _total, " - ", msg);
                  writeln("#   FAIL: ", a, " != ", b); }
  }

  proc assertEqual(a: bool, b: bool, msg: string = "") {
    _total += 1;
    if a == b { _pass += 1; writeln("ok ", _total, " - ", msg); }
    else       { _fail += 1; writeln("not ok ", _total, " - ", msg);
                  writeln("#   FAIL: ", a, " != ", b); }
  }

  proc assertNear(a: real, b: real, tol: real, msg: string = "") {
    assertEqual(a, b, tol, msg);
  }

  proc assertInRange(v: real, lo: real, hi: real, msg: string = "") {
    _total += 1;
    if v >= lo && v <= hi { _pass += 1; writeln("ok ", _total, " - ", msg); }
    else { _fail += 1; writeln("not ok ", _total, " - ", msg);
           writeln("#   FAIL: ", v, " not in [", lo, ", ", hi, "]"); }
  }

  proc assertGT(a: real, b: real, msg: string = "") {
    _total += 1;
    if a > b { _pass += 1; writeln("ok ", _total, " - ", msg); }
    else      { _fail += 1; writeln("not ok ", _total, " - ", msg);
                 writeln("#   FAIL: ", a, " <= ", b); }
  }

  proc assertLT(a: real, b: real, msg: string = "") {
    _total += 1;
    if a < b { _pass += 1; writeln("ok ", _total, " - ", msg); }
    else      { _fail += 1; writeln("not ok ", _total, " - ", msg);
                 writeln("#   FAIL: ", a, " >= ", b); }
  }
}
