"""Unit tests for text cleaning and address presence rules.

Tests the golden samples from EDA §7.1 and §7.2:
- Brackets/parentheses wrapping
- Leading punctuation runs
- Embedded URLs and domain endings
- Consecutive duplicate tokens
- Legal suffix preservation
- Invariant non-empty name enforcement
- Address presence and placeholder cleaning
"""

import unittest
from src.phase0.text_cleaner import check_has_address, clean_name_base, clean_address


class TestTextCleaner(unittest.TestCase):

    def test_bracket_paren_wrapping(self):
        self.assertEqual(clean_name_base("Sweet Barbershop (Co)"), "sweet barbershop co")
        self.assertEqual(clean_name_base("COBALT  (LLC)"), "cobalt llc")
        self.assertEqual(clean_name_base("CLASSIC [TECHNOLOGIES]"), "classic technologies")
        self.assertEqual(
            clean_name_base("[INCORPORATED] PEAK TRADIN6 NETWORKS SOUTHSIDE"),
            "incorporated peak tradin6 networks southside",
        )
        self.assertEqual(clean_name_base("AMERICAN CHOICE SERVICE [[LLC]]"), "american choice service llc")
        self.assertEqual(clean_name_base("Dubey Media {Ltd}"), "dubey media ltd")
        self.assertEqual(clean_name_base("<< Star Tech >>"), "star tech")

    def test_leading_punctuation_runs(self):
        self.assertEqual(clean_name_base("-- Holloway Peak Inc Seafood"), "holloway peak inc seafood")
        self.assertEqual(clean_name_base("*** Global Foods LLC"), "global foods llc")
        self.assertEqual(clean_name_base("~~ Horizon Enterprises"), "horizon enterprises")

    def test_url_in_name(self):
        self.assertEqual(
            clean_name_base("SHIVSHAKTI VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com"),
            "shivshakti vidyalaya overseas corporation",
        )
        self.assertEqual(
            clean_name_base("Shiva International-Pigraate Limited | www.shivainte.com"),
            "shiva international-pigraate limited",
        )
        self.assertEqual(clean_name_base("heassociates.com"), "heassociates")
        self.assertEqual(clean_name_base("prprivate.com"), "prprivate")
        self.assertEqual(clean_name_base("http://mycompany.org"), "mycompany")

    def test_repeated_consecutive_words(self):
        self.assertEqual(
            clean_name_base("Orellana Investments Investments Llc"),
            "orellana investments llc",
        )
        self.assertEqual(
            clean_name_base("SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION"),
            "shivshakti vidyalaya overseas corporation",
        )
        self.assertEqual(
            clean_name_base("राम राम प्राइवेट लिमिटेड"),
            "राम प्राइवेट लिमिटेड",
        )

    def test_legal_suffixes_preserved(self):
        # Ltd, Pvt, Inc, LLC, SARL must NOT be stripped (retrieval-time IDF concern)
        name = "Apex Solutions Private Limited"
        cleaned = clean_name_base(name)
        self.assertIn("private", cleaned)
        self.assertIn("limited", cleaned)

        french = "Thermal & Fils SARL"
        cleaned_fr = clean_name_base(french)
        self.assertIn("sarl", cleaned_fr)

    def test_non_empty_name_invariant(self):
        # A name containing only stripped symbols must fall back to non-empty
        self.assertTrue(len(clean_name_base("---")) > 0)
        self.assertTrue(len(clean_name_base("[()]")) > 0)

    def test_address_presence(self):
        self.assertTrue(check_has_address("123 Main St, Springfield, IL"))
        self.assertFalse(check_has_address(None))
        self.assertFalse(check_has_address(""))
        self.assertFalse(check_has_address("   "))
        self.assertFalse(check_has_address("NULL"))
        self.assertFalse(check_has_address("null"))
        self.assertFalse(check_has_address("<NULL>"))
        self.assertFalse(check_has_address("N/A"))
        self.assertFalse(check_has_address("n/a"))
        self.assertFalse(check_has_address("<N/A>"))
        self.assertFalse(check_has_address("nan"))

    def test_address_cleaning(self):
        # Embedded NULL removal
        addr1 = "067 PRODUCTION CT, NULL, INDEPENDENCE, KY"
        self.assertEqual(clean_address(addr1, True), "067 production ct, independence, ky")

        addr2 = "2612 SOUTHVIEW COURT, <NULL>, BRAINERD, MN"
        self.assertEqual(clean_address(addr2, True), "2612 southview court, brainerd, mn")

        # House prefix stripping (#, ##)
        addr3 = "# 123 Main St, Springfield, IL"
        self.assertEqual(clean_address(addr3, True), "123 main st, springfield, il")

        addr4 = "##45 Park Ave, New York, NY"
        self.assertEqual(clean_address(addr4, True), "45 park ave, new york, ny")

        # Absent address returns empty string
        self.assertEqual(clean_address("NULL", False), "")
        self.assertEqual(clean_address("", False), "")


if __name__ == "__main__":
    unittest.main()
