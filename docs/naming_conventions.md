# Naming Conventions

## Files and Folders

- Use lowercase `snake_case` for Python files, generated data exports, and configuration files.
- Use descriptive names with dates only when the date is analytically meaningful.
- Keep reusable logic in `src/`; notebooks should call code rather than duplicate pipeline logic.

## Data Artifacts

- Raw data should be treated as immutable.
- Processed files should include the processing stage or purpose in the filename.
- Embedding files should include the model family and date or version.

## Columns

- Use lowercase `snake_case` column names.
- Preserve original dataset columns only when needed for provenance.
- Prefix model-derived score columns with the concept being measured, such as `emotion_`, `dependency_`, or `complexity_`.

