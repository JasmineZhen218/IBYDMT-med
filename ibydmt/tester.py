import json
import logging
import os
from itertools import product
from random import shuffle
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ibydmt.classifiers import Classifier, get_predictions
from ibydmt.samplers import get_sampler
from ibydmt.testing.fdr import FDRPostProcessor
from ibydmt.testing.procedure import SKIT, SequentialTester, cSKIT, xSKIT
from ibydmt.utils.concept_data import get_dataset_with_concepts
from ibydmt.utils.config import ConceptType, Config
from ibydmt.utils.config import Constants as c
from ibydmt.utils.config import TestType, get_config
from ibydmt.utils.data import get_dataset
from ibydmt.utils.result import TestingResults

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def get_test_classes(config: Config, workdir: str = c.WORKDIR):
    test_classes_path = os.path.join(
        workdir, "results", config.name.lower(), "test_classes.txt"
    )
    if not os.path.exists(test_classes_path):
        dataset = get_dataset(config, workdir=workdir)
        classes = dataset.classes
    else:
        with open(test_classes_path, "r") as f:
            classes = [line.strip() for line in f]
    return classes


def get_local_test_idx(config: Config, workdir: str = c.WORKDIR):
    results_dir = os.path.join(workdir, "results", config.name.lower(), "local_cond")
    os.makedirs(results_dir, exist_ok=True)

    test_idx_path = os.path.join(results_dir, "local_test_idx.json")
    if not os.path.exists(test_idx_path):
        dataset = get_dataset(config, train=False)
        test_classes = get_test_classes(config, workdir=workdir)
        test_classes_idx = [
            dataset.classes.index(class_name) for class_name in test_classes
        ]

        label = np.array([l for _, l in dataset.samples])
        class_idx = {
            class_name: np.nonzero(label == class_idx)[0].tolist()
            for class_idx, class_name in zip(test_classes_idx, test_classes)
        }
        if getattr(dataset, "validate_class_idx_for_testing", None) is not None:
            validate_class_idx_for_testing = dataset.validate_class_idx_for_testing
            assert callable(validate_class_idx_for_testing)
            class_idx = validate_class_idx_for_testing(
                config, class_idx, workdir=workdir
            )

        test_idx = {
            class_name: rng.choice(
                _class_idx, config.testing.images_per_class, replace=False
            ).tolist()
            for class_name, _class_idx in class_idx.items()
        }

        with open(test_idx_path, "w") as f:
            json.dump(test_idx, f)

    with open(test_idx_path, "r") as f:
        test_idx = json.load(f)
    return test_idx


def sample_random_subset(concepts: Iterable[str], concept_idx: int, cardinality: int):
    sample_idx = list(set(range(len(concepts))) - {concept_idx})
    shuffle(sample_idx)
    return sample_idx[:cardinality]


def run_tests(config: Config, testers: Iterable[SequentialTester]):
    fdr_postprocessor = FDRPostProcessor(config)

    stop_value = len(testers) / config.testing.significance_level
    results = Parallel(n_jobs=1)(
        delayed(tester.test)(stop_on="value", stop_value=stop_value, return_wealth=True)
        for tester in testers
    )
    rejected, tau, wealths = zip(*results)
    fdr_rejected, fdr_tau = fdr_postprocessor(wealths)
    return (rejected, tau), (fdr_rejected, fdr_tau)


def test_global(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for global semantic importance of dataset"
        f" {config.data.dataset.lower()} with backbone = {config.data.backbone},"
        f" concept_type = {concept_type}, kernel = {config.testing.kernel},"
        f" kernel_scale = {config.testing.kernel_scale}, tau_max ="
        f" {config.testing.tau_max}"
    )

    predictions = get_predictions(config, workdir=workdir)

    results = TestingResults(config, "global", concept_type)

    test_classes = get_test_classes(config, workdir=workdir)
    for class_name in test_classes:
        logger.info(f"Testing class {class_name}")

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        concept_dataset = get_dataset_with_concepts(
            config, workdir=workdir, train=False, concept_class_name=concept_class_name
        )
        concepts = concept_dataset.concepts

        for _ in tqdm(range(config.testing.r)):
            testers = []
            for concept_idx, _ in enumerate(concepts):
                pi = rng.permutation(len(concept_dataset))
                Y = predictions[class_name].values[pi]
                Z = concept_dataset.semantics[:, concept_idx][pi]

                tester = SKIT(config, Y, Z)
                testers.append(tester)

            (rejected, tau), (fdr_rejected, fdr_tau) = run_tests(config, testers)
            results.add(class_name, concepts, rejected, tau)
            results.add(class_name, concepts, fdr_rejected, fdr_tau, fdr_control=True)

    results.save(workdir)


def test_global_cond(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for global conditional semantic importance of dataset"
        f" {config.data.dataset.lower()} with backbone = {config.data.backbone}, "
        f" concept_type = {concept_type}, kernel ="
        f" {config.testing.kernel}, kernel_scale = {config.testing.kernel_scale},"
        f" tau_max = {config.testing.tau_max}, ckde_scale = {config.ckde.scale}",
    )

    predictions = get_predictions(config, workdir=workdir)

    results = TestingResults(config, "global_cond", concept_type)

    test_classes = get_test_classes(config, workdir=workdir)
    for class_name in test_classes:
        logger.info(f"Testing class {class_name}")

        concept_class_name = None
        if concept_type == ConceptType.CLASS.value:
            concept_class_name = class_name

        concept_dataset = get_dataset_with_concepts(
            config, workdir=workdir, train=False, concept_class_name=concept_class_name
        )
        concepts = concept_dataset.concepts

        sampler = get_sampler(config, concept_class_name=concept_class_name)

        for _ in tqdm(range(config.testing.r)):
            testers = []
            for concept_idx, _ in enumerate(concepts):
                pi = rng.permutation(len(concept_dataset))
                Y = predictions[class_name].values[pi]
                Z = concept_dataset.semantics[pi]

                tester = cSKIT(config, Y, Z, concept_idx, sampler.sample_concept)
                testers.append(tester)

            (rejected, tau), (fdr_rejected, fdr_tau) = run_tests(config, testers)
            results.add(class_name, concepts, rejected, tau)
            results.add(class_name, concepts, fdr_rejected, fdr_tau, fdr_control=True)

    results.save(workdir)


def test_local_cond(config: Config, concept_type: str, workdir: str = c.WORKDIR):
    logger.info(
        "Testing for local conditional semantic importance of dataset"
        f" {config.data.dataset.lower()} with backbone = {config.data.backbone},"
        f" concept_type = {concept_type}, kernel = {config.testing.kernel},"
        f" kernel_scale = {config.testing.kernel_scale}, tau_max ="
        f" {config.testing.tau_max}, ckde_scale = {config.ckde.scale}"
    )

    dataset = get_dataset(config, workdir=workdir)
    classifier = Classifier.from_pretrained(config, workdir=workdir)

    test_idx = get_local_test_idx(config, workdir=workdir)
    cardinalities = config.testing.cardinality
    results = TestingResults(config, "local_cond", concept_type)

    classes = dataset.classes
    for class_name, class_test_idx in test_idx.items():
        class_idx = classes.index(class_name)

        class_test = list(product(class_test_idx, cardinalities))
        for idx, cardinality in class_test:
            logger.info(
                f"Testing id = {idx} (class = {class_name}) with cardinality ="
                f" {cardinality}"
            )

            concept_class_name = None
            concept_image_idx = None
            if concept_type == ConceptType.CLASS.value:
                concept_class_name = class_name
            if concept_type == ConceptType.IMAGE.value:
                concept_image_idx = idx

            concept_dataset = get_dataset_with_concepts(
                config,
                workdir=workdir,
                train=False,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )
            concepts = concept_dataset.concepts
            if len(concepts) > 20:
                continue

            sampler = get_sampler(
                config,
                concept_class_name=concept_class_name,
                concept_image_idx=concept_image_idx,
            )

            z = concept_dataset.semantics[idx]

            for _ in tqdm(range(config.testing.r)):
                testers = []
                for concept_idx, _ in enumerate(concepts):
                    subset_idx = sample_random_subset(
                        concepts, concept_idx, cardinality
                    )

                    tester = xSKIT(
                        config,
                        z,
                        concept_idx,
                        subset_idx,
                        sampler.sample_embedding,
                        classifier,
                        class_idx=class_idx,
                        cond_p_kwargs=dict(m=config.testing.tau_max),
                    )

                    testers.append(tester)

                (rejected, tau), (fdr_rejected, fdr_tau) = run_tests(config, testers)
                results.add(
                    class_name,
                    concepts,
                    rejected,
                    tau,
                    idx=idx,
                    cardinality=cardinality,
                )
                results.add(
                    class_name,
                    concepts,
                    fdr_rejected,
                    fdr_tau,
                    fdr_control=True,
                    idx=idx,
                    cardinality=cardinality,
                )

    results.save(workdir)


class ConceptTester(object):
    def __init__(self, config_name: str):
        self.config: Config = get_config(config_name)

    def test(self, test_type: str, concept_type: str, workdir: str = c.WORKDIR):
        if test_type == TestType.GLOBAL.value:
            test_fn = test_global
            sweep_ckde = False
        elif test_type == TestType.GLOBAL_COND.value:
            test_fn = test_global_cond
            sweep_ckde = True
        elif test_type == TestType.LOCAL_COND.value:
            test_fn = test_local_cond
            sweep_ckde = True
        else:
            raise ValueError(f"Invalid test_type: {test_type}")

        sweep_keys = [
            "data.backbone",
            "testing.kernel",
            "testing.kernel_scale",
            "testing.tau_max",
        ]
        if sweep_ckde:
            sweep_keys += ["ckde.scale"]

        for config in self.config.sweep(sweep_keys):
            test_fn(config, concept_type, workdir)
