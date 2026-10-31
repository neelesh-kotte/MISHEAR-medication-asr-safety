"""Schema checks for the released medication vocabulary format."""

import csv

from src.vocabulary import VOCABULARY_REQUIRED_COLUMNS, validate_vocabulary_csv


def _write(tmp_path, rows):
    path = tmp_path / "vocabulary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=VOCABULARY_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _valid_row(**updates):
    row = {field: "" for field in VOCABULARY_REQUIRED_COLUMNS}
    row.update({"schema_version": "1.0", "product_id": "PRODUCT-1", "drug_name": "example",
                "inn": "example", "ingredient_id": "INGREDIENT-1", "metaphone_key": "EK",
                "soundex_key": "E251", "source": "audited_source", "source_release": "2026-01"})
    row.update(updates)
    return row


def test_v1_schema_accepts_resolved_row(tmp_path):
    result = validate_vocabulary_csv(_write(tmp_path, [_valid_row()]))
    assert result.valid
    assert result.schema_version == "1.0"


def test_duplicate_product_identifier_is_rejected(tmp_path):
    result = validate_vocabulary_csv(_write(tmp_path, [_valid_row(), _valid_row(drug_name="example2")]))
    assert not result.valid
    assert any("duplicate product_id" in error for error in result.errors)


def test_alias_mapping_to_multiple_ingredients_is_rejected(tmp_path):
    second = _valid_row(product_id="PRODUCT-2", drug_name="other", ingredient_id="INGREDIENT-2", aliases="shared")
    first = _valid_row(aliases="shared")
    result = validate_vocabulary_csv(_write(tmp_path, [first, second]))
    assert not result.valid
    assert any("maps to multiple ingredient IDs" in error for error in result.errors)


def test_missing_metadata_is_rejected(tmp_path):
    row = _valid_row(inn="", source="", source_release="")
    result = validate_vocabulary_csv(_write(tmp_path, [row]))
    assert not result.valid
    assert any("INN is missing" in error for error in result.errors)
    assert any("source and source_release" in error for error in result.errors)


def test_checked_in_vocabulary_is_resolved_and_phonetic_keys_are_populated():
    result = validate_vocabulary_csv("data/vocabulary/medication_vocabulary.csv")
    assert result.valid
    assert result.row_count == 11
    import csv
    with open("data/vocabulary/medication_vocabulary.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["inn"] and row["product_id"] and row["source_release"] for row in rows)
    assert all(row["metaphone_key"] and row["soundex_key"] for row in rows)
    combination = next(row for row in rows if row["combination_ingredient_ids"])
    assert set(combination["combination_ingredient_ids"].split("|")) == {"RXNORM:6809", "RXNORM:593411"}

