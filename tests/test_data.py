"""
Some tests for the loaders_core module.
"""

import bz2
from collections import Counter
from functools import partial
import gzip
from io import BytesIO
from itertools import chain
import lzma
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zlib

if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import py7zr

from src.data.detect_packing_sorel import PackingMap, unpack
from src.data.loaders_core import (
    Materials,
    compute_integer_sizes,
    compute_float_sizes,
    tr_vl_ts_split_idx,
    tr_vl_ts_split,
    tr_vl_ts_split_idx_guarentee,
    get_bodmas_file_label_map,
    _get_sorel_file_label_map,
    get_sorel_file_label_map,
    _get_materials_clf,
    _get_materials_clf_multilabel,
    _get_materials_clf_few_shot_learning,
    _get_materials_clf_multilabel_few_shot_learning,
)
from src.data.utils import Decompressor


class TestSplitFunctions(unittest.TestCase):
    def test_compute_integer_sizes(self):
        total = 100
        self.assertEqual(compute_integer_sizes(total, 0.8, 0.1, 0.1), (80, 10, 10))
        self.assertEqual(compute_integer_sizes(total, 80, 10, 10), (80, 10, 10))

    def test_compute_float_sizes(self):
        total = 100
        self.assertEqual(compute_float_sizes(total, 0.8, 0.1, 0.1), (0.8, 0.1, 0.1))
        self.assertEqual(compute_float_sizes(total, 80, 10, 10), (0.8, 0.1, 0.1))

    def test_tr_vl_ts_split_idx(self):
        total = 100
        split_idx = tr_vl_ts_split_idx(total, 0.8, 0.1, 0.1)
        self.assertEqual(len(split_idx["tr"]), 80)
        self.assertEqual(len(split_idx["vl"]), 10)
        self.assertEqual(len(split_idx["ts"]), 10)

    def test_tr_vl_ts_split(self):
        collection = list(range(100))
        split = tr_vl_ts_split(collection, 0.8, 0.1, 0.1)
        self.assertEqual(len(split["tr"]), 80)
        self.assertEqual(len(split["vl"]), 10)
        self.assertEqual(len(split["ts"]), 10)

    def test_tr_vl_ts_split_idx_guarentee(self):
        labels = [0] * 50 + [1] * 50
        split_idx = tr_vl_ts_split_idx_guarentee(labels, 0.8, 0.1, 0.1, samples_per_class=5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["tr"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["tr"])[1], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["vl"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["vl"])[1], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["ts"])[0], 5)
        self.assertGreaterEqual(Counter(labels[i] for i in split_idx["ts"])[1], 5)


class TestDecompressor(unittest.TestCase):
    def setUp(self):
        self.test_data = b'This is a test string.'
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_file = Path(self.test_dir.name) / "test.bytes"
        with open(self.test_file, 'wb') as f:
            f.write(self.test_data)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_none_decompression(self):
        compressed_data = self.test_data
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.NONE)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.NONE)
            self.assertEqual(b, self.test_data)

    def test_gzip_decompression(self):
        compressed_data = gzip.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.GZIP)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.GZIP)
            self.assertEqual(b, self.test_data)

    def test_bzip2_decompression(self):
        compressed_data = bz2.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.BZIP2)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.BZIP2)
            self.assertEqual(b, self.test_data)

    def test_lzma_decompression(self):
        compressed_data = lzma.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.LZMA)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.LZMA)
            self.assertEqual(b, self.test_data)

    def test_zlib_decompression(self):
        compressed_data = zlib.compress(self.test_data)
        with open(self.test_file, "wb") as fp:
            fp.write(compressed_data)
        bytes_io = BytesIO(compressed_data)

        decompressor = Decompressor(Decompressor.ZLIB)
        for data in [compressed_data, self.test_file, bytes_io]:
            alg, b = decompressor(data)
            self.assertEqual(alg, Decompressor.ZLIB)
            self.assertEqual(b, self.test_data)

    @unittest.skip("Skipping test_py7zr_decompression because its not implemented yet.")
    def test_py7zr_decompression(self):
        fp = BytesIO()
        with py7zr.SevenZipFile(fp, 'w') as archive:
            archive.writef(BytesIO(self.test_data), "tmp")
        fp.seek(0)
        compressed_data = fp.read()
        decompressor = Decompressor(Decompressor.S7Z)
        alg, b = decompressor(BytesIO(compressed_data))
        self.assertEqual(alg, Decompressor.S7Z)
        self.assertEqual(b, self.test_data)


class TestPackingMap(unittest.TestCase):
    def setUp(self):
        self.maps = []

    def tearDown(self):
        self.maps = []

    def test_packing_map_0(self):
        print("packing_map_0")
        t = time.time()
        packing_map_0 = PackingMap(lazy=False, chunked=True, num_workers=16)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_0)=}")
        self.assertTrue(len(packing_map_0) > 0)
        self.maps.append(packing_map_0)

    def test_packing_map_1(self):
        print("packing_map_1")
        t = time.time()
        packing_map_1 = PackingMap(lazy=False, chunked=True, num_workers=None)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_1)=}")
        self.assertTrue(len(packing_map_1) > 0)
        self.maps.append(packing_map_1)

    def test_packing_map_2(self):
        print("packing_map_2")
        t = time.time()
        packing_map_2 = PackingMap(lazy=False, chunked=False, num_workers=None)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_2)=}")
        self.assertTrue(len(packing_map_2) > 0)
        self.maps.append(packing_map_2)

    def test_packing_map_3(self):
        print("packing_map_3")
        t = time.time()
        packing_map_3 = PackingMap(lazy=True, chunked=True, num_workers=16)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_3)=}")
        self.assertTrue(len(packing_map_3) > 0)
        self.maps.append(packing_map_3)

    @unittest.skip("Skipping test_packing_map_4 because it is eggregiously slow.")
    def test_packing_map_4(self):
        print("packing_map_4")
        t = time.time()
        packing_map_4 = PackingMap(lazy=True, chunked=False, num_workers=None)
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
        print(f"{len(packing_map_4)=}")
        self.assertTrue(len(packing_map_4) > 0)
        self.maps.append(packing_map_4)

    def test_maps_equality(self):
        for i, map1 in enumerate(self.maps):
            for j, map2 in enumerate(self.maps):
                if i != j:
                    self.assertEqual(map1, map2, f"Maps {i} and {j} are not equal")


@unittest.skip("Skipping TestUnpacking because it is not implemented.")
class TestUnpacking(unittest.TestCase):

    _test_file = "./tmp/calc.exe"

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.unpacked = self.test_dir /  "unpacked.exe"
        self.packed = self.test_dir / "packed.exe"
        self.outfile = self.test_dir / "out.exe"
        shutil.copy2(self._test_file, self.unpacked)
        args = ["upx", "--best", "-o", str(self.packed), str(self.unpacked)]
        try:
            result = subprocess.run(args, check=True, capture_output=True)
        except subprocess.CalledProcessError as err:
            print(err.stderr)
            raise err

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_packed_file(self):
        try:
            outfile, byte_0 = unpack(self.packed, self.outfile, True, True, 1)
        except subprocess.CalledProcessError as err:
            print(err.stderr)
            raise err

        byte_1 = self.unpacked.read_bytes()
        assert len(byte_0) == len(byte_1), f"{len(byte_0)=} != {len(byte_1)=}"
        # Don't know why, but the executables themselves have some small differences.

    def test_unpacked_file(self):
        with self.assertRaises(subprocess.CalledProcessError):
            unpack(self.unpacked, self.outfile, True, False, 1)
        outfile, byte = unpack(self.unpacked, self.outfile, True, True, 0)
        assert outfile is None
        assert byte is None


class GetMaterialsClf(unittest.TestCase):

    def setUp(self):
        self.file_label_map = get_bodmas_file_label_map()
        self._get_materials_clf = partial(
            _get_materials_clf,
            files_and_labels=self.file_label_map,
            tr_size=0.8,
            vl_size=0.1,
            ts_size=0.1,
            must_exist=False,
        )

    def _test_materials_object(self, materials: Materials) -> None:
        tr_classes = set(materials.labels["tr"])
        vl_classes = set(materials.labels["vl"])
        ts_classes = set(materials.labels["ts"])
        assert tr_classes == vl_classes == ts_classes, f"{len(tr_classes)=} {len(vl_classes)=} {len(ts_classes)=}"

    def test_top_k(self):
        materials = self._get_materials_clf(top_k=10)
        assert materials.num_classes == 10
        self._test_materials_object(materials)

    def test_min_freq(self):
        materials = self._get_materials_clf(min_freq=100)
        assert all(v >= 100 for v in materials.dist.values())
        self._test_materials_object(materials)

    def test_max_imbalance_ratio(self):
        materials_a = self._get_materials_clf(max_imbalance_ratio=None)
        materials_b = self._get_materials_clf(max_imbalance_ratio=sys.maxsize)
        materials_c = self._get_materials_clf(max_imbalance_ratio=100)

        self._test_materials_object(materials_a)
        self._test_materials_object(materials_b)
        self._test_materials_object(materials_c)

        c_a = set(materials_a.dist.keys())
        c_b = set(materials_b.dist.keys())
        c_c = set(materials_c.dist.keys())

        assert len(c_a.difference(c_b)) == 0, f"{c_a.difference(c_b)=}"
        assert len(c_a.difference(c_c)) == 0, f"{c_a.difference(c_c)=}"
        assert len(c_b.difference(c_a)) == 0, f"{c_b.difference(c_a)=}"
        assert len(c_b.difference(c_c)) == 0, f"{c_b.difference(c_c)=}"
        assert len(c_c.difference(c_a)) == 0, f"{c_c.difference(c_a)=}"
        assert len(c_c.difference(c_b)) == 0, f"{c_c.difference(c_b)=}"

        assert materials_c.dist.most_common(1)[0][1] <= 100 * materials_c.dist.most_common()[-1][1]


class GetMaterialsClfMultilabel(unittest.TestCase):

    def setUp(self):
        self.file_label_map = get_sorel_file_label_map("beh")
        self._get_materials_clf_multilabel = partial(
            _get_materials_clf_multilabel,
            files_and_labels=self.file_label_map,
            tr_size=0.8,
            vl_size=0.1,
            ts_size=0.1,
            must_exist=False,
        )

    def _test_materials_object(self, materials: Materials) -> None:
        tr_classes = set(chain.from_iterable(materials.labels["tr"]))
        vl_classes = set(chain.from_iterable(materials.labels["vl"]))
        ts_classes = set(chain.from_iterable(materials.labels["ts"]))
        assert tr_classes == vl_classes == ts_classes, f"{len(tr_classes)=} {len(vl_classes)=} {len(ts_classes)=}"

    def test_top_k(self):
        materials = self._get_materials_clf_multilabel(top_k=10)
        assert materials.num_classes == 10
        self._test_materials_object(materials)

    def test_min_freq(self):
        materials = self._get_materials_clf_multilabel(min_freq=100)
        assert all(v >= 100 for v in materials.dist.values())
        self._test_materials_object(materials)

    def test_max_imbalance_ratio(self):
        materials_a = self._get_materials_clf_multilabel(max_imbalance_ratio=None)
        materials_b = self._get_materials_clf_multilabel(max_imbalance_ratio=sys.maxsize)
        materials_c = self._get_materials_clf_multilabel(max_imbalance_ratio=100)

        self._test_materials_object(materials_a)
        self._test_materials_object(materials_b)
        self._test_materials_object(materials_c)

        c_a = set(materials_a.dist.keys())
        c_b = set(materials_b.dist.keys())
        c_c = set(materials_c.dist.keys())

        assert len(c_a.difference(c_b)) == 0, f"{c_a.difference(c_b)=}"
        assert len(c_a.difference(c_c)) == 0, f"{c_a.difference(c_c)=}"
        assert len(c_b.difference(c_a)) == 0, f"{c_b.difference(c_a)=}"
        assert len(c_b.difference(c_c)) == 0, f"{c_b.difference(c_c)=}"
        assert len(c_c.difference(c_a)) == 0, f"{c_c.difference(c_a)=}"
        assert len(c_c.difference(c_b)) == 0, f"{c_c.difference(c_b)=}"

        # The filtering method is approximate in nature and min_freq takes precsedence, so we need some tolerance.
        assert materials_c.dist_tr.most_common(1)[0][1] <= 10 * 100 * materials_c.dist_tr.most_common()[-1][1], f"{materials_c.dist_tr=}"
        assert materials_c.dist_vl.most_common(1)[0][1] <= 10 * 100 * materials_c.dist_vl.most_common()[-1][1], f"{materials_c.dist_vl=}"
        assert materials_c.dist_ts.most_common(1)[0][1] <= 10 * 100 * materials_c.dist_ts.most_common()[-1][1], f"{materials_c.dist_ts=}"


class TestGetMaterialsClfFewShotLearning(unittest.TestCase):

    def setUp(self):
        self.files_and_labels = get_bodmas_file_label_map()
        self.tr_samples_per_class = list(range(1, 10))

    def _test_materials(
        self,
        materials: Materials,
        tr_samples_per_class: int,
        vl_min_samples_per_class: int,
        vl_max_samples_per_class,
    ) -> None:
        # print(f"{tr_samples_per_class=}\n{materials}\n{'-' * 80}")
        print(f"{tr_samples_per_class=} {len(materials.dist)=}")
        n_files = len(materials.files["tr"]) + len(materials.files["vl"])
        n_unique_files = len(set(materials.files["tr"] + materials.files["vl"]))
        assert n_files == n_unique_files, f"{n_files=} != {n_unique_files=}"
        assert all(v == tr_samples_per_class for v in materials.dist_tr.values()), f"{materials.dist_tr=}"
        assert all(vl_min_samples_per_class <= v <= vl_max_samples_per_class for v in materials.dist_vl.values()), f"{materials.dist_vl=}"
        assert set(materials.dist.keys()) == (set(materials.dist_tr.keys())) == (set(materials.dist_vl.keys()))

    def test_one(self):
        vl_min_samples_per_class = 1
        vl_max_samples_per_class = 10
        for tr_samples_per_class in self.tr_samples_per_class:
            materials = _get_materials_clf_few_shot_learning(
                self.files_and_labels,
                tr_samples_per_class,
                vl_min_samples_per_class=vl_min_samples_per_class,
                vl_max_samples_per_class=vl_max_samples_per_class,
                top_k=None,
            )
            self._test_materials(materials, tr_samples_per_class, vl_min_samples_per_class, vl_max_samples_per_class)

    def test_two(self):
        vl_min_samples_per_class = 4
        vl_max_samples_per_class = 20
        for tr_samples_per_class in self.tr_samples_per_class:
            materials = _get_materials_clf_few_shot_learning(
                self.files_and_labels,
                tr_samples_per_class,
                vl_min_samples_per_class=vl_min_samples_per_class,
                vl_max_samples_per_class=vl_max_samples_per_class,
                top_k=None,
            )
            self._test_materials(materials, tr_samples_per_class, vl_min_samples_per_class, vl_max_samples_per_class)


@unittest.skip("Skipping TestGetMaterialsClfMultilabelFewShotLearning because it is not complete.")
class TestGetMaterialsClfMultilabelFewShotLearning(unittest.TestCase):

    def setUp(self):
        self.files_and_labels = get_sorel_file_label_map("beh")
        self.tr_samples_per_class = list(range(1, 10))

    def _test_materials(
        self,
        materials: Materials,
        tr_samples_per_class: int,
        tr_max_samples_per_class: int,
        vl_min_samples_per_class: int,
        vl_max_samples_per_class: int,
    ) -> None:
        info = f"{len(materials.dist)=} {tr_samples_per_class=} {tr_max_samples_per_class=} {vl_min_samples_per_class=} {vl_max_samples_per_class=} "
        n_files = len(materials.files["tr"]) + len(materials.files["vl"])
        n_unique_files = len(set(materials.files["tr"] + materials.files["vl"]))
        assert n_files == n_unique_files, info + f"{n_files=} != {n_unique_files=}"
        assert all(tr_samples_per_class <= v <= tr_max_samples_per_class for v in materials.dist_tr.values()), info + f"{materials.dist_tr=}"
        assert all(vl_min_samples_per_class <= v <= vl_max_samples_per_class for v in materials.dist_vl.values()), info + f"{materials.dist_vl=}"
        assert set(materials.dist.keys()) == (set(materials.dist_tr.keys())) == (set(materials.dist_vl.keys())), info + f"{len(materials.dist)=} {len(materials.dist_tr)=} {len(materials.dist_vl)=}"

    def test_one(self):
        vl_min_samples_per_class = 1
        for tr_samples_per_class in self.tr_samples_per_class:
            materials = _get_materials_clf_multilabel_few_shot_learning(
                self.files_and_labels,
                tr_samples_per_class,
                vl_min_samples_per_class=vl_min_samples_per_class,
                vl_max_samples_per_class=None,
                top_k=None,
            )
            self._test_materials(materials, tr_samples_per_class, 20 * tr_samples_per_class, vl_min_samples_per_class, sys.maxsize)

    def test_two(self):
        vl_min_samples_per_class = 4
        for tr_samples_per_class in self.tr_samples_per_class:
            materials = _get_materials_clf_multilabel_few_shot_learning(
                self.files_and_labels,
                tr_samples_per_class,
                vl_min_samples_per_class=vl_min_samples_per_class,
                vl_max_samples_per_class=None,
                top_k=None,
            )
            self._test_materials(materials, tr_samples_per_class, 20 * tr_samples_per_class, vl_min_samples_per_class, sys.maxsize)


class Test_GetSorelFileLabelMap(unittest.TestCase):

    def setUp(self):
        self.files_and_labels = _get_sorel_file_label_map()

    def get_sorel_file_label_map(self, name: str):
        files_and_labels = {f: getattr(l, name) for f, l in self.files_and_labels.items()}
        files_and_labels = {f: l for f, l in files_and_labels.items() if l is not None}
        return files_and_labels

    def test_file_label_map(self, files_and_labels: dict, single_label: bool):
        for file, label in files_and_labels.items():
            self.assertIsInstance(file, (os.PathLike, Path, str))
            self.assertIsInstance(label, tuple)
            if single_label:
                self.assertEqual(len(label), 1)
            for l in label:
                l: str
                self.assertIsInstance(l, str), f"{l=}"
                assert not l.isspace(), f"{l=}"
                assert not l.lower() in ("none", "na", "nan"), f"{l=}"

    def test_fam(self):
        files_and_labels = self.get_sorel_file_label_map("fam")
        self.test_file_label_map(files_and_labels, single_label=True)

    def test_file(self):
        files_and_labels = self.get_sorel_file_label_map("file")
        self.test_file_label_map(files_and_labels, single_label=True)

    def test_class_(self):
        files_and_labels = self.get_sorel_file_label_map("class_")
        self.test_file_label_map(files_and_labels, single_label=False)

    def test_beh(self):
        files_and_labels = self.get_sorel_file_label_map("beh")
        self.test_file_label_map(files_and_labels, single_label=False)

    def test_pack(self):
        files_and_labels = self.get_sorel_file_label_map("pack")
        self.test_file_label_map(files_and_labels, single_label=False)

    @unittest.skip("Skipping test_unk because it is not implemented.")
    def test_unk(self):
        files_and_labels = self.get_sorel_file_label_map("unk")
        self.test_file_label_map(files_and_labels, single_label=False)

    @unittest.skip("Skipping test_vuln because it is not implemented.")
    def test_vuln(self):
        files_and_labels = self.get_sorel_file_label_map("vuln")
        self.test_file_label_map(files_and_labels, single_label=False)


if __name__ == "__main__":
    unittest.main()
