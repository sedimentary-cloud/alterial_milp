"""标准错误类型：编号、原因/建议、与内置异常的兼容性。"""
from __future__ import annotations

import unittest

from greenwave import (GWError, GWInfeasibleGridPoint, GWInputError,
                       GWInvariantError, GWModelError, GWSemanticError,
                       GWSolveError, GWValidationError, ValidationIssue)
from greenwave.schema.models import AtomRef


class TestStandardErrors(unittest.TestCase):
    def test_input_error_keeps_valueerror_compatibility(self):
        exc = GWInputError("bad unit", hint="unit 只能是 s/ratio")
        self.assertIsInstance(exc, ValueError)
        self.assertIsInstance(exc, GWError)
        self.assertEqual(exc.code, "GW-INPUT")
        self.assertIn("可能原因", str(exc))
        self.assertIn("建议", str(exc))
        self.assertEqual(exc.message, "bad unit")

    def test_invariant_error_keeps_assertion_compatibility(self):
        exc = GWInvariantError("x is None")
        self.assertIsInstance(exc, AssertionError)
        self.assertEqual(exc.code, "GW-INTERNAL")
        self.assertTrue(exc.hint)

    def test_real_schema_failure_uses_gw_input_error(self):
        with self.assertRaises(GWInputError) as ctx:
            AtomRef.from_obj({"kind": "boundary"})
        text = str(ctx.exception)
        self.assertIn("[GW-INPUT]", text)
        self.assertIn("missing fields", text)
        self.assertIn("建议", text)

    def test_model_and_solve_errors_have_default_guidance(self):
        for exc in (GWModelError("bad ref"), GWSolveError("solver down")):
            self.assertTrue(exc.cause)
            self.assertTrue(exc.hint)
            self.assertIn(exc.code, str(exc))

    def test_error_registry_codes_are_unique_and_all_subclass_gwerror(self):
        classes = [GWInputError, GWInvariantError, GWValidationError,
                   GWInfeasibleGridPoint, GWModelError, GWSemanticError,
                   GWSolveError]
        codes = [c.code for c in classes]
        self.assertEqual(len(codes), len(set(codes)))
        for cls in classes:
            self.assertTrue(issubclass(cls, GWError))
        self.assertEqual(
            set(GWInputError("x").to_dict()),
            {"code", "message", "cause", "hint", "context"},
        )

    def test_validation_and_grid_errors_keep_structured_context(self):
        issue = ValidationIssue("error", "V1", "bad field", ("x",))
        ve = GWValidationError([issue])
        self.assertEqual(ve.issues, (issue,))
        self.assertIn("V1", str(ve))
        self.assertEqual(ve.context["issues"], ["V1"])

        ge = GWInfeasibleGridPoint(90.0, 1.0, {"I1": ("no plan",)})
        self.assertEqual(ge.cycle_s, 90.0)
        self.assertIn("I1", str(ge))
        self.assertIn("建议", str(ge))


if __name__ == "__main__":
    unittest.main()
