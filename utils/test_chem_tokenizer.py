import unittest
from chem_tokenizer import get_elements


class TestChemTokenizer(unittest.TestCase):
    def test_rules(self):
        """Testing Chris Konop's rules"""
        tests = (
            ("Fe3 O", "Fe3O"),
            ("Zr5 Al", "Zr5Al"),
            ("Mn 4+", "Mn4+"),
            ("Na 2", "Na2"),
            ("Fe3O 4", "Fe3O4"),
            ("Mn 0.75", "Mn0.75"),
            ("Zn3 PO", "Zn3PO"),
        )

        for input, expected in tests:
            with self.subTest(msg=f"Testing {input} -> {expected}"):
                self.assertEqual(get_elements(input), {expected})

    def advanced_tests(self):
        tests = (("Fe-Cr-O-H-inert", "FeCrOH"),)

        for input, expected in tests:
            with self.subTest(msg=f"Testing {input} -> {expected}"):
                self.assertEqual(get_elements(input), {expected})

        # Implement behavior: YBa 2 Cu 3 S 6-y => YBa2Cu3S6-y


if __name__ == "__main__":
    unittest.main(verbosity=2)
