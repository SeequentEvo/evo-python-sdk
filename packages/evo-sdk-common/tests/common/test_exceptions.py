#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import copy
import platform
import sys
import traceback
import unittest
from collections.abc import Callable, Sequence

from evo.common.exceptions import BadRequestException, EvoExceptionGroup


class TestInvalidObjectError(unittest.TestCase):
    def test_validation_paths_and_full_detail_are_preserved(self) -> None:
        content = {
            "type": "https://seequent.com/error-codes/geoscienceobject/schemas/invalid-object",
            "title": "The Geoscience Object Is Invalid.",
            "detail": f"{{'values': '{'x' * 1000}'}} is not valid under any of the given schemas",
            "json_path": "$.collections[0]",
            "schema_path": ["properties", "collections", "items", "oneOf"],
        }
        error = BadRequestException(400, "Bad Request", content, None)
        message = str(error)
        self.assertIn("Invalid value at: $.collections[0]", message)
        self.assertIn(f"Failed schema rule: {content['schema_path']}", message)
        self.assertIn("is not valid under any of the given schemas", message)
        self.assertNotIn("x" * 1000, message)
        self.assertIn("Full message available in .detail", message)
        self.assertEqual(content["detail"], error.detail)
        self.assertIs(content, error.content)
        self.assertIs(type(error), BadRequestException)

    def test_missing_paths_and_detail_preview_boundaries(self) -> None:
        for length in (499, 500, 501):
            with self.subTest(length=length):
                detail = "x" * length
                content = {
                    "type": "https://seequent.com/error-codes/geoscienceobject/schemas/invalid-object",
                    "title": "Invalid object",
                    "detail": detail,
                }
                message = str(BadRequestException(400, "Bad Request", content, None))
                self.assertNotIn("Invalid value at:", message)
                self.assertNotIn("Failed schema rule:", message)
                self.assertEqual(length > 500, "[omitted]" in message)
                if length <= 500:
                    self.assertIn(f"Detail: {detail}", message)

    def test_unrelated_bad_requests_are_unchanged(self) -> None:
        content = {"title": "Bad request", "detail": "x" * 1000, "json_path": "$.unrelated"}
        error = BadRequestException(400, "Bad Request", content, None)
        self.assertEqual(
            f"Error: (400) Bad Request\nType: about:blank\nTitle: Bad request\nDetail: {content['detail']}",
            str(error),
        )


class MyExc1(Exception): ...


class MyExc2(Exception): ...


class MyExc3(MyExc1): ...


MY_EXC_TYPES = (MyExc1, MyExc2, MyExc3)


def tb_text(exc: Exception) -> str:
    return "".join(traceback.format_exception(exc))


class TestEvoExceptionGroup(unittest.TestCase):
    EGROUP_TYPE = EvoExceptionGroup

    @classmethod
    def filter_excs(
        cls, excs: Sequence[Exception, ...], filter_func: Callable[[Exception], bool], inverse: bool = False
    ) -> tuple[Exception, ...]:
        flattened = []
        for exc in excs:
            if isinstance(exc, cls.EGROUP_TYPE):
                matched, unmatched = exc.split(filter_func)
                if matched and not inverse:
                    flattened.append(matched)
                    continue
                elif unmatched and inverse:
                    flattened.append(unmatched)
                    continue
            if filter_func(exc) ^ inverse:
                flattened.append(exc)
        return tuple(flattened)

    def raise_excs(self, excs: list[Exception]) -> EGROUP_TYPE:
        with self.assertRaises(self.EGROUP_TYPE) as cm:
            raise self.EGROUP_TYPE("Some exceptions occurred", excs)
        return cm.exception

    def compare_excs(self, expected: Exception, actual: Exception) -> None:
        if isinstance(expected, self.EGROUP_TYPE) or isinstance(actual, self.EGROUP_TYPE):
            self.assertEqual(expected.message, actual.message)
            self.assertEqual(expected.__traceback__, actual.__traceback__)
            self.assertEqual(expected.__context__, actual.__context__)
            self.assertEqual(expected.__cause__, actual.__cause__)
            self.assertEqual(
                len(expected.exceptions),
                len(actual.exceptions),
                f"Number of exceptions differ\n{tb_text(expected)} != {tb_text(actual)}",
            )
            for sub_expected, sub_actual in zip(expected.exceptions, actual.exceptions):
                self.compare_excs(sub_expected, sub_actual)
        else:
            self.assertEqual(expected, actual)

    def check_derived(self, parent: EGROUP_TYPE, derived: EGROUP_TYPE, expected_exceptions: tuple[Exception, ...]):
        self.assertIsInstance(derived.exceptions, tuple)

        expected = parent.derive(expected_exceptions)
        # Copy the original context, cause, and traceback.
        expected.__context__ = copy.copy(parent.__context__)
        expected.__cause__ = copy.copy(parent.__cause__)
        expected.__traceback__ = copy.copy(parent.__traceback__)

        self.compare_excs(expected, derived)

    def test_single_wrapped_exception(self) -> None:
        excs = [MyExc1("Single exception")]
        grp = self.raise_excs(excs)

        with self.subTest("field: message"):
            self.assertEqual("Some exceptions occurred", grp.message)

        with self.subTest("field: exceptions"):
            self.assertTupleEqual(tuple(excs), grp.exceptions)

        with self.subTest("method: subgroup"):
            self.check_derived(grp, grp.subgroup(MyExc1), tuple(excs))
            self.assertIsNone(grp.subgroup(MyExc2))
            self.check_derived(grp, grp.subgroup(self.EGROUP_TYPE), grp.exceptions)

        with self.subTest("method: split"):
            matched, unmatched = grp.split(MyExc1)
            self.check_derived(grp, matched, tuple(excs))
            self.assertIsNone(unmatched)

            matched, unmatched = grp.split(self.EGROUP_TYPE)
            self.check_derived(grp, matched, grp.exceptions)
            self.assertIsNone(unmatched)

    def test_multiple_same_exception(self) -> None:
        excs = [MyExc1(f"Multiple same exception {n}") for n in range(10)]
        grp = self.raise_excs(excs)

        with self.subTest("field: exceptions"):
            self.assertTupleEqual(tuple(excs), grp.exceptions)

        with self.subTest("method: subgroup"):
            self.check_derived(grp, grp.subgroup(MyExc1), tuple(excs))
            self.assertIsNone(grp.subgroup(MyExc2))
            self.check_derived(grp, grp.subgroup(self.EGROUP_TYPE), grp.exceptions)

        with self.subTest("method: split"):
            matched, unmatched = grp.split(MyExc1)
            self.check_derived(grp, matched, tuple(excs))
            self.assertIsNone(unmatched)

            matched, unmatched = grp.split(self.EGROUP_TYPE)
            self.check_derived(grp, matched, grp.exceptions)
            self.assertIsNone(unmatched)

    def test_multiple_different_exceptions(self) -> None:
        excs = []
        for n in range(10):
            msg = f"Multiple different exceptions {n}"
            exc_type = MY_EXC_TYPES[n % 2]
            excs.append(exc_type(msg))
        grp = self.raise_excs(excs)

        with self.subTest("field: exceptions"):
            self.assertTupleEqual(tuple(excs), grp.exceptions)

        tuple_myexc1 = self.filter_excs(excs, lambda e: isinstance(e, MyExc1))
        tuple_myexc2 = self.filter_excs(excs, lambda e: isinstance(e, MyExc2))
        tuple_msg = self.filter_excs(excs, lambda e: str(e) == msg)
        tuple_not_msg = self.filter_excs(excs, lambda e: str(e) != msg)

        with self.subTest("method: subgroup"):
            self.check_derived(grp, grp.subgroup(MyExc1), tuple_myexc1)
            self.check_derived(grp, grp.subgroup(MyExc2), tuple_myexc2)
            self.check_derived(grp, grp.subgroup((MyExc1, MyExc2)), tuple(excs))
            self.check_derived(grp, grp.subgroup(lambda e: str(e) == msg), tuple_msg)
            self.check_derived(grp, grp.subgroup(self.EGROUP_TYPE), grp.exceptions)

        with self.subTest("method: split"):
            matched, unmatched = grp.split(MyExc1)
            self.check_derived(grp, matched, tuple_myexc1)
            self.check_derived(grp, unmatched, tuple_myexc2)

            matched, unmatched = grp.split(MyExc2)
            self.check_derived(grp, matched, tuple_myexc2)
            self.check_derived(grp, unmatched, tuple_myexc1)

            matched, unmatched = grp.split((MyExc1, MyExc2))
            self.check_derived(grp, matched, tuple(excs))
            self.assertIsNone(unmatched)

            matched, unmatched = grp.split(lambda e: str(e) == msg)
            self.check_derived(grp, matched, tuple_msg)
            self.check_derived(grp, unmatched, tuple_not_msg)

            matched, unmatched = grp.split(self.EGROUP_TYPE)
            self.check_derived(grp, matched, grp.exceptions)
            self.assertIsNone(unmatched)

    def test_multiple_different_exceptions_incl_subclass(self) -> None:
        excs = []
        for n in range(10):
            msg = f"Multiple different exceptions including subclasses {n}"
            exc_type = MY_EXC_TYPES[n % 2]
            excs.append(exc_type(msg))
        grp = self.raise_excs(excs)

        with self.subTest("field: exceptions"):
            self.assertTupleEqual(tuple(excs), grp.exceptions)

        tuple_myexc1_myexc3 = self.filter_excs(excs, lambda e: isinstance(e, (MyExc1, MyExc3)))
        tuple_myexc2 = self.filter_excs(excs, lambda e: isinstance(e, MyExc2))
        tuple_msg = self.filter_excs(excs, lambda e: str(e) == msg)
        tuple_not_msg = self.filter_excs(excs, lambda e: str(e) == msg, inverse=True)

        with self.subTest("method: subgroup"):
            self.check_derived(grp, grp.subgroup(MyExc1), tuple_myexc1_myexc3)
            self.check_derived(grp, grp.subgroup(MyExc2), tuple_myexc2)
            self.check_derived(grp, grp.subgroup((MyExc1, MyExc2)), tuple(excs))
            self.check_derived(grp, grp.subgroup(lambda e: str(e) == msg), tuple_msg)
            self.check_derived(grp, grp.subgroup(self.EGROUP_TYPE), grp.exceptions)

        with self.subTest("method: split"):
            matched, unmatched = grp.split(MyExc1)
            self.check_derived(grp, matched, tuple_myexc1_myexc3)
            self.check_derived(grp, unmatched, tuple_myexc2)

            matched, unmatched = grp.split(MyExc2)
            self.check_derived(grp, matched, tuple_myexc2)
            self.check_derived(grp, unmatched, tuple_myexc1_myexc3)

            matched, unmatched = grp.split((MyExc1, MyExc2))
            self.check_derived(grp, matched, tuple(excs))
            self.assertIsNone(unmatched)

            matched, unmatched = grp.split(lambda e: str(e) == msg)
            self.check_derived(grp, matched, tuple_msg)
            self.check_derived(grp, unmatched, tuple_not_msg)

            matched, unmatched = grp.split(self.EGROUP_TYPE)
            self.check_derived(grp, matched, grp.exceptions)
            self.assertIsNone(unmatched)

    def test_subgroups(self) -> None:
        excs = []
        for i in range(10):
            subgrp_excs = []
            for j in range(i + 1):
                exc_msg = f"Exc {i + 1}.{j + 1}"
                exc_type = MY_EXC_TYPES[(i * j) % 3]
                subgrp_excs.append(exc_type(exc_msg))
            excs.append(self.EGROUP_TYPE(f"Subgroup {i + 1}", subgrp_excs))

        grp = self.raise_excs(excs)

        with self.subTest("field: exceptions"):
            self.assertTupleEqual(tuple(excs), grp.exceptions)

        tuple_myexc1_myexc3 = self.filter_excs(excs, lambda e: isinstance(e, (MyExc1, MyExc3)))
        tuple_myexc2 = self.filter_excs(excs, lambda e: isinstance(e, MyExc2))
        tuple_msg = self.filter_excs(excs, lambda e: str(e) == exc_msg)
        tuple_not_msg = self.filter_excs(excs, lambda e: str(e) == exc_msg, inverse=True)

        with self.subTest("method: subgroup"):
            self.check_derived(grp, grp.subgroup(MyExc1), tuple_myexc1_myexc3)
            self.check_derived(grp, grp.subgroup(MyExc2), tuple_myexc2)
            self.check_derived(grp, grp.subgroup((MyExc1, MyExc2)), tuple(excs))
            self.check_derived(grp, grp.subgroup(lambda e: str(e) == exc_msg), tuple_msg)
            self.check_derived(grp, grp.subgroup(self.EGROUP_TYPE), grp.exceptions)

        with self.subTest("method: split"):
            matched, unmatched = grp.split(MyExc1)
            self.check_derived(grp, matched, tuple_myexc1_myexc3)
            self.check_derived(grp, unmatched, tuple_myexc2)

            matched, unmatched = grp.split(MyExc2)
            self.check_derived(grp, matched, tuple_myexc2)
            self.check_derived(grp, unmatched, tuple_myexc1_myexc3)

            matched, unmatched = grp.split((MyExc1, MyExc2))
            self.check_derived(grp, matched, tuple(excs))
            self.assertIsNone(unmatched)

            matched, unmatched = grp.split(lambda e: str(e) == exc_msg)
            self.check_derived(grp, matched, tuple_msg)
            self.check_derived(grp, unmatched, tuple_not_msg)

            matched, unmatched = grp.split(self.EGROUP_TYPE)
            self.check_derived(grp, matched, grp.exceptions)
            self.assertIsNone(unmatched)

    def test_nested_subgroups(self) -> None:
        excs = []
        for i in range(10):
            subgrp_i_excs = []
            for j in range(i + 1):
                subgrp_j_excs = []
                for k in range(j + 1):
                    exc_msg = f"Exc {i + 1}.{j + 1}.{k + 1}"
                    exc_type = MY_EXC_TYPES[(i * j) % 3]
                    subgrp_j_excs.append(exc_type(exc_msg))
                subgrp_i_excs.append(self.EGROUP_TYPE(f"Subgroup {i + 1}.{j + 1}", subgrp_j_excs))
            excs.append(self.EGROUP_TYPE(f"Subgroup {i + 1}", subgrp_i_excs))

        grp = self.raise_excs(excs)

        with self.subTest("field: exceptions"):
            self.assertTupleEqual(tuple(excs), grp.exceptions)

        tuple_myexc1_myexc3 = self.filter_excs(excs, lambda e: isinstance(e, (MyExc1, MyExc3)))
        tuple_myexc2 = self.filter_excs(excs, lambda e: isinstance(e, MyExc2))
        tuple_msg = self.filter_excs(excs, lambda e: str(e) == exc_msg)
        tuple_not_msg = self.filter_excs(excs, lambda e: str(e) == exc_msg, inverse=True)

        with self.subTest("method: subgroup"):
            self.check_derived(grp, grp.subgroup(MyExc1), tuple_myexc1_myexc3)
            self.check_derived(grp, grp.subgroup(MyExc2), tuple_myexc2)
            self.check_derived(grp, grp.subgroup((MyExc1, MyExc2)), tuple(excs))
            self.check_derived(grp, grp.subgroup(lambda e: str(e) == exc_msg), tuple_msg)
            self.check_derived(grp, grp.subgroup(self.EGROUP_TYPE), grp.exceptions)

        with self.subTest("method: split"):
            matched, unmatched = grp.split(MyExc1)
            self.check_derived(grp, matched, tuple_myexc1_myexc3)
            self.check_derived(grp, unmatched, tuple_myexc2)

            matched, unmatched = grp.split(MyExc2)
            self.check_derived(grp, matched, tuple_myexc2)
            self.check_derived(grp, unmatched, tuple_myexc1_myexc3)

            matched, unmatched = grp.split((MyExc1, MyExc2))
            self.check_derived(grp, matched, tuple(excs))
            self.assertIsNone(unmatched)

            matched, unmatched = grp.split(lambda e: str(e) == exc_msg)
            self.check_derived(grp, matched, tuple_msg)
            self.check_derived(grp, unmatched, tuple_not_msg)

            matched, unmatched = grp.split(self.EGROUP_TYPE)
            self.check_derived(grp, matched, grp.exceptions)
            self.assertIsNone(unmatched)


@unittest.skipUnless(
    sys.version_info >= (3, 11), f"ExceptionGroup is not implemented in python {platform.python_version()}"
)
class TestBuiltinExceptionGroup(TestEvoExceptionGroup):
    """Run the same tests on the builtin ExceptionGroup type when it is available"""

    @classmethod
    def setUpClass(cls) -> None:
        # Defer this assignment until runtime to prevent an exception in earlier versions of python.
        cls.EGROUP_TYPE = ExceptionGroup  # noqa: F821
