import time

import numpy as np
import torch
import datasets
from tqdm import tqdm


class HiddenStateLoader:
    """Vendorized loader from Interlat for HF datasets of stored hidden states."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.id_to_data = {}
        self._load_data()

    def _load_data(self) -> None:
        print(f"Loading tensor data from {self.dataset_name}")
        self.dataset = datasets.load_dataset(self.dataset_name, split=datasets.Split.TRAIN)
        print(f"Loaded {len(self.dataset)} records.")

        def optimized_convert_nested_arrays_with_plan(df):
            print(f"Optimized converting {len(df)} nested arrays with plan text...")
            start_time = time.time()

            def optimized_nested_convert(nested_array):
                try:
                    if isinstance(nested_array, np.ndarray) and nested_array.dtype == object:
                        list_data = nested_array.tolist()
                        numpy_array = np.array(list_data, dtype=np.float32)
                        return torch.from_numpy(numpy_array)
                    return torch.from_numpy(nested_array.astype(np.float32))
                except Exception as exc:  # pragma: no cover - diagnostic vendor behavior
                    print(f"Conversion failed: {exc}")
                    return None

            df["tensor_hidden_state"] = df["hidden_state"].apply(optimized_nested_convert)
            success_mask = df["tensor_hidden_state"].notna()
            success_count = int(success_mask.sum())
            print(f"Successfully converted: {success_count}/{len(df)} arrays")

            valid_df = df[success_mask]
            id_to_data = {}
            for _, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Building id_to_data"):
                id_to_data[row["task_id"]] = {
                    "hidden_state": row["tensor_hidden_state"],
                    "plan": row["plan"],
                }

            conversion_time = time.time() - start_time
            print(f"Optimized conversion completed in {conversion_time:.2f} seconds")
            if id_to_data:
                sample_key = next(iter(id_to_data))
                sample_data = id_to_data[sample_key]
                print(f"Sample tensor shape: {tuple(sample_data['hidden_state'].shape)}")
                print(f"Sample tensor dtype: {sample_data['hidden_state'].dtype}")
                print(f"Sample plan: {sample_data['plan'][:100]}...")
            return id_to_data

        with tqdm(total=1, desc="Converting Dataset to Pandas") as pbar:
            df = self.dataset.to_pandas()
            pbar.update(1)
        self.id_to_data = optimized_convert_nested_arrays_with_plan(df)

    def get_hidden_state_and_plan(self, task_id):
        if task_id not in self.id_to_data:
            raise KeyError(f"No hidden_state found for task_id: {task_id}")
        item = self.id_to_data[task_id]
        return item["hidden_state"], item["plan"]
