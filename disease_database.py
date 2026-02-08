#!/usr/bin/env python3

"""
Disease Treatment Database Module
===================================
Manages the plant disease treatment database and provides access to
disease information, symptoms, and treatment recommendations.

Educational Project - SunFounder Electronic Kit
"""

import json
import os

class TreatmentDatabase:
    """
    Treatment Database class for managing plant disease information.
    Loads treatment data from JSON file and provides lookup methods.
    """

    def __init__(self, data_file='data/treatments.json'):
        """
        Initialize the treatment database.

        Args:
            data_file (str): Path to the treatments JSON file
        """
        self.data_file = data_file
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.data_dir, data_file)
        self.treatments = {}
        self.loaded = False

    def load(self):
        """
        Load treatment data from JSON file.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.data_path, 'r') as f:
                self.treatments = json.load(f)
            self.loaded = True
            print(f"✓ Loaded {len(self.treatments)} disease treatments from database")
            return True
        except FileNotFoundError:
            print(f"✗ Error: Treatment database not found at {self.data_path}")
            self.loaded = False
            return False
        except json.JSONDecodeError as e:
            print(f"✗ Error: Invalid JSON in treatment database: {e}")
            self.loaded = False
            return False
        except Exception as e:
            print(f"✗ Error loading treatment database: {e}")
            self.loaded = False
            return False

    def get_treatment(self, class_label):
        """
        Get treatment information for a specific disease class.

        Args:
            class_label (str): The disease class label (e.g., "Tomato___Early_blight")

        Returns:
            dict: Treatment information or None if not found
        """
        if not self.loaded:
            print("Warning: Database not loaded. Call load() first.")
            return None

        return self.treatments.get(class_label, None)

    def get_common_name(self, class_label):
        """
        Get the common/display name for a disease.

        Args:
            class_label (str): The disease class label

        Returns:
            str: Common name or the class label if not found
        """
        treatment = self.get_treatment(class_label)
        if treatment:
            return treatment.get('common_name', class_label)
        return class_label

    def get_all_diseases(self):
        """
        Get a list of all diseases in the database.

        Returns:
            list: List of disease class labels
        """
        return list(self.treatments.keys())

    def get_disease_count(self):
        """
        Get the total number of diseases in the database.

        Returns:
            int: Number of diseases
        """
        return len(self.treatments)

    def format_treatment_info(self, class_label):
        """
        Format treatment information as a human-readable string.

        Args:
            class_label (str): The disease class label

        Returns:
            str: Formatted treatment information
        """
        treatment = self.get_treatment(class_label)

        if not treatment:
            return f"No information available for: {class_label}"

        # Build formatted string
        output = []
        output.append("=" * 60)
        output.append(f"DIAGNOSIS: {treatment.get('common_name', 'Unknown')}")
        output.append("=" * 60)

        output.append(f"\nPlant: {treatment.get('plant', 'Unknown')}")
        output.append(f"Disease: {treatment.get('disease', 'Unknown')}")
        output.append(f"Severity: {treatment.get('severity', 'Unknown').upper()}")

        output.append(f"\nDescription:")
        output.append(f"  {treatment.get('description', 'No description available.')}")

        # Symptoms
        symptoms = treatment.get('symptoms', [])
        if symptoms:
            output.append(f"\nSymptoms:")
            for symptom in symptoms:
                output.append(f"  • {symptom}")

        # Treatments
        treatments = treatment.get('treatments', [])
        if treatments:
            output.append(f"\nTreatment Recommendations:")
            for idx, treat in enumerate(treatments, 1):
                output.append(f"  {idx}. {treat}")

        # Prevention
        prevention = treatment.get('prevention', [])
        if prevention:
            output.append(f"\nPrevention Measures:")
            for prev in prevention:
                output.append(f"  • {prev}")

        output.append("\n" + "=" * 60)
        output.append("NOTE: This is for educational purposes only.")
        output.append("      Consult agricultural extension services for professional advice.")
        output.append("=" * 60)

        return "\n".join(output)

    def is_healthy(self, class_label):
        """
        Check if the classification indicates a healthy plant.

        Args:
            class_label (str): The disease class label

        Returns:
            bool: True if healthy, False otherwise
        """
        return 'healthy' in class_label.lower()


# Test the database module
def test_database():
    """
    Test function for the treatment database.
    """
    print("========================================")
    print("|    Treatment Database Test           |")
    print("========================================\n")

    # Create and load database
    db = TreatmentDatabase()

    if not db.load():
        print("Failed to load database!")
        return

    print(f"\nTotal diseases in database: {db.get_disease_count()}")

    # Test with a few diseases
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
