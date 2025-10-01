import pandas as pd
from datetime import datetime

class ElectricityDataLoader:
    """
    Class to handle loading and preprocessing of electricity dataset.
    """

    def __init__(self, filepath: str):
        """
        Initialize the loader with the dataset path.

        Args:
            filepath (str): Path to the CSV file.
        """
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from the provided file path.

        Returns:
            pd.DataFrame: Raw loaded DataFrame.
        """
        self.df = pd.read_csv(self.filepath)
        return self.df

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the loaded dataset:
        - Parse datetime
        - Add derived columns (DATE, TIME, YEAR, MONTH, DAY)

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_data()` first.")

        # Convert to datetime
        self.df["SETTLEMENTDATE"] = pd.to_datetime(
            self.df["SETTLEMENTDATE"], format='mixed', errors='coerce'
        )

        # Create datetime-related features
        self.df["DATE"] = self.df["SETTLEMENTDATE"].dt.date
        self.df["TIME"] = self.df["SETTLEMENTDATE"].dt.time
        self.df["YEAR"] = self.df["SETTLEMENTDATE"].dt.year
        self.df["MONTH"] = self.df["SETTLEMENTDATE"].dt.month
        self.df["DAY"] = self.df["SETTLEMENTDATE"].dt.day

        return self.df

    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Convenience method: load and preprocess the dataset in one step.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        self.load_data()
        return self.preprocess()


    def compute_anchor_now_iso(
        self,
        *,
        tz: str = "Australia/Sydney",
        source_tz: str = "Australia/Sydney",
    ) -> str:
        """
        Return the latest SETTLEMENTDATE as an ISO-8601 string in `tz`.

        - If SETTLEMENTDATE is tz-naive, it is localized to `source_tz`.
        - Then converted to `tz` and the maximum timestamp is returned.

        Returns:
            str: ISO-8601 datetime string (e.g., '2024-09-01T12:30:00+10:00')
        """
        if self.df is None or self.df.empty or "SETTLEMENTDATE" not in self.df.columns:
            raise ValueError("Data not loaded/preprocessed or missing SETTLEMENTDATE column.")

        s = pd.to_datetime(self.df["SETTLEMENTDATE"], errors="coerce", utc=False)
        if s.isna().all():
            raise ValueError("No valid SETTLEMENTDATE values after parsing.")

        # Localize if tz-naive; then convert
        try:
            if getattr(s.dt, "tz", None) is None:
                s = s.dt.tz_localize(source_tz, nonexistent="shift_forward", ambiguous="NaT")
            s = s.dt.tz_convert(tz)
        except Exception:
            # Fallback: try to coerce to timezone-aware via UTC then convert
            s = s.dt.tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert(tz)

        mx = s.max()
        if pd.isna(mx):
            raise ValueError("Unable to compute latest timestamp from SETTLEMENTDATE.")
        return mx.isoformat()