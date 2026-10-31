"""Tiny deterministic smoke checks for vocabulary candidate ranking."""

import unittest

from src.vocabulary import (
    MedicationName,
    generate_candidate_pairs,
    jaro_winkler,
    load_vocabulary_csv,
    phonetic_similarity,
    soundex,
)


class VocabularyPilotTests(unittest.TestCase):
    def test_exact_and_near_string_similarity(self):
        self.assertEqual(jaro_winkler("", ""), 1.0)
        self.assertGreater(jaro_winkler("hydroxyzine", "hydralazine"), 0.70)
        self.assertGreaterEqual(phonetic_similarity("hydroxyzine", "hydralazine"), 0.66)
        self.assertEqual(soundex("Robert"), soundex("Rupert"))

    def test_generates_example_candidate_but_excludes_same_ingredient(self):
        vocabulary = [
            MedicationName("hydroxyzine", "1", "ingredient-1", "IN"),
            MedicationName("hydralazine", "2", "ingredient-2", "IN"),
            MedicationName("hydroxyzine hydrochloride", "3", "ingredient-1", "SCD"),
            MedicationName("atorvastatin", "4", "ingredient-4", "IN"),
        ]
        pairs = generate_candidate_pairs(vocabulary)
        names = {(pair.name_a, pair.name_b) for pair in pairs}
        self.assertIn(("hydroxyzine", "hydralazine"), names)
        flagged = next(pair for pair in pairs if (pair.name_a, pair.name_b) == ("hydroxyzine", "hydralazine"))
        self.assertGreaterEqual(flagged.phonetic_similarity, 0.66)
        self.assertTrue(flagged.high_similarity_candidate)
        self.assertNotIn(("hydroxyzine", "hydroxyzine hydrochloride"), names)

    def test_loads_csv_aliases(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "terms.csv"
            path.write_text("STR,RXCUI,ingredient_id,TTY\nDrug A,10,1,IN\n", encoding="utf-8")
            rows = load_vocabulary_csv(path)
        self.assertEqual(rows[0].name, "Drug A")
        self.assertEqual(rows[0].ingredient_rxcui, "1")


if __name__ == "__main__":
    unittest.main()
