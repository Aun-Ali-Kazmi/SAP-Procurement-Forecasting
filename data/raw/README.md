# Raw data (not included in the repository)

The analysis uses a purchase-order line export from SAP S/4HANA
(`final_set.xlsx`, sheet `Dataset for Budget forecast (2)`). It contains
confidential company procurement data and is therefore **not** published.

To reproduce the results, place the export here as:

    data/raw/final_set.xlsx

Required columns: `Plant`, `Material`, `MATERIAL DETAILS`, `Document Date`,
`Net Order Value`. Optional columns used for the exploratory figures:
`Deletion indicator`, `Still to be delivered (qty)`, `Currency`.
