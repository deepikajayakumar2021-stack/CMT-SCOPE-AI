import os
import numpy as np
import pandas as pd


# STAGE 5B: VERIFY GAIT DATA AGAINST PATIENT MANIFEST


PROJECT_ROOT = r"D:\CMT_PROJECT"

MANIFEST_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "stage5_patient_manifest.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "splits",
    "stage5b_gait_verification.csv"
)



# LOAD MANIFEST

manifest = pd.read_csv(MANIFEST_FILE)

print("=" * 70)
print("STAGE 5B: GAIT DATA VERIFICATION")
print("=" * 70)

print("\nPatients in manifest:", len(manifest))



# VERIFY EACH PATIENT


results = []

for _, row in manifest.iterrows():

    patient_id = row["ID"]
    gait_file = row["gait_file"]

    result = {
        "ID": patient_id,
        "split": row["split"],
        "file_exists": False,
        "X_shape": None,
        "y_shape": None,
        "patient_id_inside_file": None,
        "nan_count": None,
        "inf_count": None,
        "valid": False
    }

    try:

        if not os.path.isfile(gait_file):

            print(
                f"\nWARNING: Gait file not found for {patient_id}"
            )

            results.append(result)
            continue

        result["file_exists"] = True

        # ----------------------------------------------------
        # LOAD NPZ
        # ----------------------------------------------------

        data = np.load(
            gait_file,
            allow_pickle=True
        )

        print("\n" + "-" * 70)
        print("Patient:", patient_id)
        print("File:", os.path.basename(gait_file))
        print("Keys:", data.files)

        

        if "X" not in data.files:

            print("ERROR: X missing")
            results.append(result)
            continue

        X = data["X"]

        result["X_shape"] = str(X.shape)

        print("X shape:", X.shape)

        if "y" in data.files:

            y = data["y"]

            result["y_shape"] = str(y.shape)

            print("y shape:", y.shape)

            print(
                "Unique activity labels:",
                np.unique(y)
            )

        else:

            print("WARNING: y missing")

         
        if "patient_id" in data.files:

            stored_id = data["patient_id"]

            if np.ndim(stored_id) == 0:

                stored_id = stored_id.item()

            stored_id = str(stored_id)

            result[
                "patient_id_inside_file"
            ] = stored_id

            print(
                "Patient ID inside file:",
                stored_id
            )

            if stored_id != patient_id:

                print(
                    "ERROR: Patient ID mismatch!"
                )

                results.append(result)
                continue

        else:

            print(
                "WARNING: patient_id missing"
            )

        # ----------------------------------------------------
        # NUMERICAL CHECKS
        # ----------------------------------------------------

        nan_count = int(
            np.isnan(X).sum()
        )

        inf_count = int(
            np.isinf(X).sum()
        )

        result["nan_count"] = nan_count
        result["inf_count"] = inf_count

        print(
            "NaN count:",
            nan_count
        )

        print(
            "Inf count:",
            inf_count
        )

        # ----------------------------------------------------
        # SHAPE CHECK
        # ----------------------------------------------------

        shape_valid = (
            X.ndim == 3 and
            X.shape[1] == 256 and
            X.shape[2] == 3
        )

        if not shape_valid:

            print(
                "ERROR: Unexpected X shape"
            )

            results.append(result)
            continue

        # ----------------------------------------------------
        # NUMERICAL VALIDITY
        # ----------------------------------------------------

        if (
            nan_count == 0
            and
            inf_count == 0
        ):

            result["valid"] = True

            print(
                "STATUS: VALID"
            )

        else:

            print(
                "STATUS: INVALID NUMERICAL VALUES"
            )

        results.append(result)

    except Exception as e:

        print(
            "ERROR:",
            e
        )

        results.append(result)



verification = pd.DataFrame(
    results
)

verification.to_csv(
    OUTPUT_FILE,
    index=False
)
 

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(
    "\nTotal patients:",
    len(verification)
)

print(
    "Valid patients:",
    int(verification["valid"].sum())
)

print(
    "Invalid patients:",
    int((~verification["valid"]).sum())
)

print(
    "Files found:",
    int(verification["file_exists"].sum())
)

print("\nShapes:")

print(
    verification[
        "X_shape"
    ].value_counts()
)

print("\nWindows per patient:")

print(
    verification[
        ["ID", "split", "X_shape"]
    ].to_string(index=False)
)

print("\nSaved:")

print(
    OUTPUT_FILE
)

print("\nSTAGE 5B COMPLETE")