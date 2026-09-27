# Loader Module Specification (`data_layer/loader`)

## 1. Description
This module is responsible for loading the massive raw `.tsv` files natively using the Polars Rust-backed engine. It acts as the first line of defense against messy data by standardizing fake string nulls and optimizing memory footprints before any intensive string processing occurs.

## 2. EARS Requirements

### Ubiquitous Requirements
*   **REQ-LDR-01**: The `loader` module shall use `polars` to read the input TSV files, leveraging multi-threading to prevent OOM errors.
*   **REQ-LDR-02**: The `loader` module shall cast the `country` column to a Categorical data type to drastically reduce memory usage.

### Event-Driven Requirements
*   **REQ-LDR-03**: When reading a TSV file, the module shall parse string literals that represent nulls (e.g., `"n/a"`, `"na"`, `"null"`, `"none"`, `"-"`) and convert them into actual empty strings `""`.

### State-Driven Requirements
*   **REQ-LDR-04**: While reading the file, the module shall immediately impute any actual missing values (Polars Nulls) with empty strings `""` to prevent `NoneType` errors in downstream string pipelines.

### Unwanted Behavior Requirements
*   **REQ-LDR-05**: If the input file is not a valid TSV, the module shall raise an `InvalidFileFormatError`.

## 3. Schemas

### Input Schema
*   **filepath**: `str` (Path to the raw TSV file)

### Output Schema
*   **dataframe**: `polars.DataFrame`
    *   `entity_id`: String
    *   `business_name`: String (Guaranteed no nulls, only empty strings)
    *   `business_address`: String (Guaranteed no nulls, only empty strings)
    *   `country`: Categorical

## 4. Errors & Exceptions
*   `InvalidFileFormatError`: Raised if the file is missing or fails to parse as a tab-separated value file.
