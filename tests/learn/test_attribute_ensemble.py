"""
Test.
"""

from pathlib import Path
import tempfile
import os
import unittest

import torch
from torch import Tensor
import transformers

from src.utils import ignore_warnings_decorator, print_context
from src.architectures.hrrformer import HRREnsembleForSequenceClassification, HRRConfig
try:
    from src.learn.attribute_ensemble import Attributor, AttributorRunner
    IS_CAPTUM_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as _err:
    if _err.name != "captum":
        raise _err
    IS_CAPTUM_AVAILABLE = False
from src.learn.collators import EnsembleDataCollatorWithPadding
from src.tokenization.api import get_fast_tokenizer


ENABLE_UNITTEST_WARNING = os.environ.get("LMLM_ENABLE_UNITTEST_WARNING", "0") == "1"
ENABLE_UNITTEST_LOGGING = os.environ.get("LMLM_ENABLE_UNITTEST_LOGGING", "0") == "1"

if not ENABLE_UNITTEST_WARNING:
    transformers.logging.set_verbosity_error()

IGNORE_WARNINGS_FILTER_ACTION = "default" if ENABLE_UNITTEST_WARNING else "ignore"


ignore_layer_integrated_gradients = ignore_warnings_decorator(IGNORE_WARNINGS_FILTER_ACTION, category=UserWarning, message=r"^Multiple layers provided*")


@unittest.skipIf(not IS_CAPTUM_AVAILABLE, "Skipping TestAttributor.")
class TestAttributor(unittest.TestCase):

    def setUp(self):
        self.config = HRRConfig(vocab_size=256, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, num_labels=2)
        self.model = HRREnsembleForSequenceClassification(self.config)
        self.dataset = self.get_dataset(8, 16)
        self.raw_tokenizer = get_fast_tokenizer("raw", "wdl", 8, vocab_size=self.config.vocab_size)
        self.dis_tokenizer = get_fast_tokenizer("dis", "wdl", 8, vocab_size=self.config.vocab_size)
        self.dec_tokenizer = get_fast_tokenizer("dec", "wdl", 8, vocab_size=self.config.vocab_size)
        self.data_collator = EnsembleDataCollatorWithPadding(self.raw_tokenizer, self.dis_tokenizer, self.dec_tokenizer)

    def get_dataset(self, n_samples: int, max_length: int) -> list[dict[str, Tensor]]:
        dataset = []
        for _ in range(n_samples):
            d = {
                "labels": torch.randint(0, 2, (1,)),
                "raw_input_ids": torch.randint(0, self.config.vocab_size, (1, torch.randint(3, max_length, ()))),
                "dis_input_ids": torch.randint(0, self.config.vocab_size, (1, torch.randint(3, max_length, ()))),
                "dec_input_ids": torch.randint(0, self.config.vocab_size, (1, torch.randint(3, max_length, ()))),
            }
            dataset.append(d)
        return dataset

    def check_runner_outfile(self, file: Path, null: bool = False):
        d = torch.load(file)
        for k in ("labels", "attribs_raw", "attribs_dis", "attribs_dec"):
            assert k in d, f"Expected key '{k}' in output, got {d.keys()}"
            if null:
                assert d[k] is None or isinstance(d[k], Tensor), f"Expected {k} to be Optional[Tensor], got {type(d[k])}"
            else:
                assert isinstance(d[k], Tensor), f"Expected {k} to be Tensor, got {type(d[k])}"

    @ignore_layer_integrated_gradients
    def test_attributor(self):
        attributor = Attributor(self.model, self.dataset, self.data_collator, self.raw_tokenizer.pad_token_id, "cpu")
        with print_context(suppress=not ENABLE_UNITTEST_LOGGING):
            for step, labels, raw_input_ids, dis_input_ids, dec_input_ids in attributor:
                assert isinstance(step, int), f"Expected step to be int, got {type(step)}"
                assert isinstance(labels, Tensor), f"Expected labels to be Tensor, got {type(labels)}"
                assert isinstance(raw_input_ids, Tensor), f"Expected raw_input_ids to be Tensor, got {type(raw_input_ids)}"
                assert isinstance(dis_input_ids, Tensor), f"Expected dis_input_ids to be Tensor, got {type(dis_input_ids)}"
                assert isinstance(dec_input_ids, Tensor), f"Expected dec_input_ids to be Tensor, got {type(dec_input_ids)}"
                assert labels.dim() == 0, f"Expected labels to have dim 0, got {labels.dim()}"
                assert raw_input_ids.dim() == 1, f"Expected raw_input_ids to have dim 1, got {raw_input_ids.dim()}"
                assert dis_input_ids.dim() == 1, f"Expected dis_input_ids to have dim 1, got {dis_input_ids.dim()}"
                assert dec_input_ids.dim() == 1, f"Expected dec_input_ids to have dim 1, got {dec_input_ids.dim()}"

    @ignore_layer_integrated_gradients
    def test_runner_1(self):
        attributor = Attributor(self.model, self.dataset, self.data_collator, self.raw_tokenizer.pad_token_id, "cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            runner = AttributorRunner(tmpdir, attributor, resume=False, rerun=False, clean=False)
            with print_context(suppress=not ENABLE_UNITTEST_LOGGING):
                runner()

            for f in tmpdir.iterdir():
                self.check_runner_outfile(f)

    @ignore_layer_integrated_gradients
    def test_runner_2(self):
        attributor = Attributor(self.model, self.dataset, self.data_collator, self.raw_tokenizer.pad_token_id, "cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            test_file_1 = tmpdir / "step-00000.pt"
            test_file_2 = tmpdir / "step-00006.pt"
            test_data_1 = "TEST1"
            test_data_2 = "TEST2"
            torch.save(test_data_1, test_file_1)
            torch.save(test_data_2, test_file_2)

            runner = AttributorRunner(tmpdir, attributor, resume=True, rerun=False, clean=False)
            with print_context(suppress=not ENABLE_UNITTEST_LOGGING):
                runner()

            for f in runner.output.iterdir():
                if f in (test_file_1, test_file_2):
                    continue
                self.check_runner_outfile(f)

            data = torch.load(test_file_1)
            assert data == test_data_1, f"Expected {test_file_1} to contain '{test_data_1}', got `{data}`."
            data = torch.load(test_file_2)
            assert data == test_data_2, f"Expected {test_file_2} to contain '{test_data_2}', got `{data}`."

    @ignore_layer_integrated_gradients
    def test_runner_3(self):
        attributor = Attributor(self.model, self.dataset, self.data_collator, self.raw_tokenizer.pad_token_id, "cpu")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            test_file_1 = tmpdir / "step-00000.pt"
            test_file_2 = tmpdir / "step-00006.pt"
            test_data_1 = {"attribs_raw": "not None"}
            test_data_2 = {"attribs_raw": None}
            torch.save(test_data_1, test_file_1)
            torch.save(test_data_2, test_file_2)

            runner = AttributorRunner(tmpdir, attributor, resume=True, rerun=True, clean=False)
            with print_context(suppress=not ENABLE_UNITTEST_LOGGING):
                runner()

            for f in runner.output.iterdir():
                if f in (test_file_1, test_file_2):
                    continue
                self.check_runner_outfile(f)

            data = torch.load(test_file_1)
            assert data == test_data_1, f"Expected {test_file_1} to contain '{test_data_1}', got `{data}`."
            data = torch.load(test_file_2)
            assert set(data.keys()) == {"labels", "attribs_raw", "attribs_dis", "attribs_dec"}, f"Expected {test_file_2} to contain attributions, got `{data}`."
