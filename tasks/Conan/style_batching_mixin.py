from tasks.Conan.style_sampling import build_condition_balanced_dataloader
from utils.commons.hparams import hparams


class ConanStyleBatchingMixin:
    def _use_style_balanced_sampling(self, dataset, shuffle):
        if not shuffle or not hparams.get("use_style_balanced_sampling", False):
            return False
        return hasattr(dataset, "get_local_condition_ids") or hasattr(dataset, "datasets")

    def build_dataloader(
        self,
        dataset,
        shuffle,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=-1,
        endless=False,
        batch_by_size=True,
    ):
        if not self._use_style_balanced_sampling(dataset, shuffle):
            return super().build_dataloader(
                dataset,
                shuffle,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
                endless=endless,
                batch_by_size=batch_by_size,
            )
        trainer = getattr(self, "trainer", None)
        use_ddp = bool(getattr(trainer, "use_ddp", False))
        return build_condition_balanced_dataloader(
            dataset,
            condition_field=hparams.get("style_balanced_field", "style"),
            shuffle=shuffle,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            endless=endless,
            batch_by_size_enabled=batch_by_size,
            bucket_size=hparams.get("style_balanced_bucket_size", 0),
            bucket_factor=hparams.get("style_balanced_bucket_factor", 8),
            group_size=hparams.get("style_balanced_group_size", 2),
            include_unlabeled=hparams.get("style_balanced_include_unlabeled", True),
            use_ddp=use_ddp,
        )
