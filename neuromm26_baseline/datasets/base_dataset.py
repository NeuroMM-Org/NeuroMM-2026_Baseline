"""Base dataset definition."""


class BaseDataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError
