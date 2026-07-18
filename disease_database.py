#!/usr/bin/env python3

"""
Disease Treatment Database Module
===================================
Manages the plant disease treatment database and provides access to
disease information, symptoms, and treatment recommendations.

Educational Project - SunFounder Electronic Kit
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class TreatmentDatabase:
    """
    Treatment Database class for managing plant disease information.
    Loads treatment data from JSON file and provides lookup methods.
    """

    def __init__(self, data_file: str = 'data/treatments.json') -> None:
        """
        Initialize the treatment database.

        Args:
            data_file: Path to the treatments JSON file, relative to this file.
        """
        self.data_file = data_file
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.data_dir, data_file)
        self.treatments: dict[str, dict[str, Any]] = {}
        self.loaded = False

    def load(self) -> bool:
        """
        Load treatment data from JSON file.

        Returns:
            True if successful, False otherwise.
        """
        try:
            with open(self.data_path, 'r') as f:
                self.treatments = json.load(f)
            self.loaded = True
            logger.info("✓ Loaded %d disease treatments from database", len(self.treatments))
            return True
        except FileNotFoundError:
            logger.error("✗ Treatment database not found at %s", self.data_path)
        except json.JSONDecodeError as e:
            logger.error("✗ Invalid JSON in treatment database: %s", e)
        except Exception:
            logger.exception("✗ Error loading treatment database")
        self.loaded = False
        return False

    def get_treatment(self, class_label: str) -> Optional[dict[str, Any]]:
        """
        Get treatment information for a specific disease class.

        Args:
            class_label: The disease class label (e.g., "Tomato___Early_blight").

        Returns:
            Treatment information dict, or None if not found.
        """
        if not self.loaded:
            logger.warning("Database not loaded. Call load() first.")
            return None

        return self.treatments.get(class_label)

    def get_common_name(self, class_label: str) -> str:
        """
        Get the common/display name for a disease.

        Args:
            class_label: The disease class label.

        Returns:
            Common name, or the class label if not found.
        """
        treatment = self.get_treatment(class_label)
        if treatment:
            return treatment.get('common_name', class_label)
        return class_label

    def get_all_diseases(self) -> list[str]:
        """Get a list of all disease class labels in the database."""
        return list(self.treatments.keys())

    def get_disease_count(self) -> int:
        """Get the total number of diseases in the database."""
        return len(self.treatments)

    def format_treatment_info(self, class_label: str) -> str:
        """
        Format treatment information as a human-readable string.

        Args:
            class_label: The disease class label.

        Returns:
            Formatted treatment information.
        """
        treatment = self.get_treatment(class_label)

        if not treatment:
            return f"No information available for: {class_label}"

        output = []
        output.append("=" * 60)
        output.append(f"DIAGNOSIS: {treatment.get('common_name', 'Unknown')}")
        output.append("=" * 60)

        output.append(f"\nPlant: {treatment.get('plant', 'Unknown')}")
        output.append(f"Disease: {treatment.get('disease', 'Unknown')}")
        output.append(f"Severity: {treatment.get('severity', 'Unknown').upper()}")

        output.append("\nDescription:")
        output.append(f"  {treatment.get('description', 'No description available.')}")

        symptoms = treatment.get('symptoms', [])
        if symptoms:
            output.append("\nSymptoms:")
            for symptom in symptoms:
                output.append(f"  • {symptom}")

        treatments = treatment.get('treatments', [])
        if treatments:
            output.append("\nTreatment Recommendations:")
            for idx, treat in enumerate(treatments, 1):
                output.append(f"  {idx}. {treat}")

        prevention = treatment.get('prevention', [])
        if prevention:
            output.append("\nPrevention Measures:")
            for prev in prevention:
                output.append(f"  • {prev}")

        output.append("\n" + "=" * 60)
        output.append("NOTE: This is for educational purposes only.")
        output.append("      Consult agricultural extension services for professional advice.")
        output.append("=" * 60)

        return "\n".join(output)

    def is_healthy(self, class_label: str) -> bool:
        """Check if the classification indicates a healthy plant."""
        return 'healthy' in class_label.lower()


# Test the database module
def test_database() -> None:
    """
    Test function for the treatment database.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("========================================")
    print("|    Treatment Database Test           |")
    print("========================================\n")

    db = TreatmentDatabase()

    if not db.load():
        print("Failed to load database!")
        return

    print(f"\nTotal diseases in database: {db.get_disease_count()}")

    test_cases = [
        "Tomato___Late_blight",
        "Apple___healthy",
        "Potato___Early_blight",
        "Invalid_Disease"
    ]

    for disease in test_cases:
        print(f"\n--- Testing: {disease} ---")
        print(f"Common name: {db.get_common_name(disease)}")
        print(f"Is healthy: {db.is_healthy(disease)}")

        if disease == "Tomato___Late_blight":
            print("\nFull treatment info:")
            print(db.format_treatment_info(disease))


if __name__ == '__main__':
    test_database()
