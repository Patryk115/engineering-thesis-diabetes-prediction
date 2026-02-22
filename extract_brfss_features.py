import pandas as pd

in_file = "LLCP2023.XPT"
out_file = "LLCP2023_subset.csv"

features = [
    'DIABETE4',
    '_RFHYPE6',
    'TOLDHI3',
    '_CHOLCH3',
    '_BMI5',
    'SMOKE100',
    'CVDSTRK3',
    '_MICHD',
    '_TOTINDA',
    '_RFDRHV8',
    '_HLTHPL1',
    'MEDCOST1',
    'GENHLTH',
    'MENTHLTH',
    'PHYSHLTH',
    'DIFFWALK',
    'SEXVAR',
    '_AGEG5YR',
    'EDUCA',
    'INCOME3'
]

def extract_selected_features():

    sas_reader = pd.read_sas(
        in_file,
        chunksize=100000,
        iterator=True,
        format='xport'
    )

    found_columns = []
    is_first_chunk = True

    for i, fragment in enumerate(sas_reader):
        if is_first_chunk:
            all_columns = list(fragment.columns)
            found_columns = [col for col in features if col in all_columns]
            missing_columns = [col for col in features if col not in all_columns]

            if missing_columns:
                print(f"W pliku źródłowym nie znaleziono następujących kolumn: {missing_columns}")
                print("Zostaną one pominięte w procesie ekstrakcji")

            subset = fragment[found_columns]

            subset.to_csv( out_file, index=False, mode='w', header=True)
            is_first_chunk = False

        else:
            subset = fragment[found_columns]
            subset.to_csv(out_file,index=False,mode='a',header=False)

    print(f"Plik został zapisany jako: {out_file}")

if __name__ == "__main__":
    extract_selected_features()