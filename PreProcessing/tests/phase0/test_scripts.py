"""Unit tests for open-set script detection and transliteration registry."""

import unittest
from src.phase0.script_detector import detect_script
from src.phase0.transliteration import transliterate_name


class TestScriptDetectorAndTransliteration(unittest.TestCase):

    def test_latin_script_detection(self):
        script, is_cross = detect_script("Apex Solutions LLC")
        self.assertEqual(script, "Latin")
        self.assertFalse(is_cross)

        # Accented characters in French
        script_fr, is_cross_fr = detect_script("Société Générale de Distribution")
        self.assertEqual(script_fr, "Latin")
        self.assertFalse(is_cross_fr)

    def test_indic_script_detection(self):
        script_hi, is_cross_hi = detect_script("राम मां लॉजिस्टिक्स")
        self.assertEqual(script_hi, "Devanagari")
        self.assertTrue(is_cross_hi)

        script_ta, is_cross_ta = detect_script("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்")
        self.assertEqual(script_ta, "Tamil")
        self.assertTrue(is_cross_ta)

        script_te, is_cross_te = detect_script("నవంబర్ ఎంటర్‌ప్రైజెస్")
        self.assertEqual(script_te, "Telugu")
        self.assertTrue(is_cross_te)

        script_kn, is_cross_kn = detect_script("ಕರ್ನಾಟಕ ಎಂಟರ್ಪ್ರೈಸಸ್")
        self.assertEqual(script_kn, "Kannada")
        self.assertTrue(is_cross_kn)

    def test_mixed_script_detection(self):
        # Mixed Indic + English: should be flagged as cross-script with dominant non-Latin script
        script_bg, is_cross_bg = detect_script("এসএস Services")
        self.assertTrue(is_cross_bg)
        self.assertEqual(script_bg, "Bengali")

        script_hi, is_cross_hi = detect_script("राम Logistics Private Limited")
        self.assertTrue(is_cross_hi)
        self.assertEqual(script_hi, "Devanagari")

    def test_open_set_non_indic_script_detection(self):
        # Arabic
        script_ar, is_cross_ar = detect_script("شركة الاتحاد للتجارة")
        self.assertEqual(script_ar, "Arabic")
        self.assertTrue(is_cross_ar)

        # Cyrillic
        script_ru, is_cross_ru = detect_script("Северная Торговая Компания")
        self.assertEqual(script_ru, "Cyrillic")
        self.assertTrue(is_cross_ru)

        # CJK
        script_zh, is_cross_zh = detect_script("上海恒达实业有限公司")
        self.assertEqual(script_zh, "CJK")
        self.assertTrue(is_cross_zh)

    def test_transliteration_indic_optitrans(self):
        # Indic scripts must be transliterated and normalized
        t_name, handled, degenerate = transliterate_name(
            "राम मां लॉजिस्टिक्स", "Devanagari", True
        )
        self.assertTrue(handled)
        self.assertFalse(degenerate)
        self.assertIn("rama", t_name)
        self.assertIn("loajistiksa", t_name)
        self.assertFalse(any(ord(c) > 127 for c in t_name))

    def test_transliteration_tamil_nnna_fix(self):
        # Tamil NNNA (ன, U+0BA9) must not leak untransliterated
        t_name, handled, degenerate = transliterate_name(
            "குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்", "Tamil", True
        )
        self.assertTrue(handled)
        self.assertFalse(degenerate)
        self.assertNotIn("\u0ba9", t_name)
        self.assertIn("bhijhinas", t_name)
        self.assertFalse(any(ord(c) > 127 for c in t_name))

    def test_transliteration_malayalam_chillus_fix(self):
        # Malayalam chillu letters (ൺ ൻ ർ ൽ ൾ ൿ) must not leak
        t_name, handled, degenerate = transliterate_name(
            "അൽ കൺസ്ട്രക്ഷൻസ് ഫുഡ്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്", "Malayalam", True
        )
        self.assertTrue(handled)
        self.assertFalse(degenerate)
        for chillu in ["\u0d7a", "\u0d7b", "\u0d7c", "\u0d7d", "\u0d7e", "\u0d7f", "\u0d57"]:
            self.assertNotIn(chillu, t_name)
        self.assertFalse(any(ord(c) > 127 for c in t_name))

    def test_transliteration_bengali_nukta_fix(self):
        # Bengali nukta (়, U+09BC) must not survive as lone diacritic
        t_name, handled, degenerate = transliterate_name(
            "ডায়নামিক ইন্ডাস্ট্রিজ প্রাইভেট লিমিটেড", "Bengali", True
        )
        self.assertTrue(handled)
        self.assertFalse(degenerate)
        self.assertNotIn("\u09bc", t_name)
        self.assertFalse(any(ord(c) > 127 for c in t_name))

    def test_transliteration_telugu_zwnj_fix(self):
        # Telugu Zero-Width Non-Joiner (ZWNJ, U+200C) must be stripped
        t_name, handled, degenerate = transliterate_name(
            "నవంబర్ ఎంటర్‌ప్రైజెస్", "Telugu", True
        )
        self.assertTrue(handled)
        self.assertFalse(degenerate)
        self.assertNotIn("\u200c", t_name)
        self.assertFalse(any(ord(c) > 127 for c in t_name))

    def test_transliteration_unhandled_script_passthrough(self):
        # Non-Indic scripts pass through untouched and record handled=False
        ar_name = "شركة الاتحاد للتجارة"
        t_name, handled, degenerate = transliterate_name(ar_name, "Arabic", True)
        self.assertFalse(handled)
        self.assertFalse(degenerate)
        self.assertEqual(t_name, ar_name)

    def test_latin_passthrough(self):
        # Latin rows must pass through unchanged
        lat_name = "apex solutions llc"
        t_name, handled, degenerate = transliterate_name(lat_name, "Latin", False)
        self.assertFalse(handled)
        self.assertFalse(degenerate)
        self.assertEqual(t_name, lat_name)


if __name__ == "__main__":
    unittest.main()
