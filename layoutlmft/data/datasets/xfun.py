# Lint as: python3
import json
import logging
import os

import datasets

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer

_URL = os.path.join(os.getcwd(), "/work/Datasets/Doc-understanding/invoice_data/xfund-format/")
print(_URL)

_LANG = ["zh", "de", "es", "fr", "en", "it", "ja", "pt"]
logger = logging.getLogger(__name__)

labels = ['cnt', 'price', 'name', 'company', 'date', 'total']
ner_labels = ['O'] + [pre + '-' + label for label in labels for pre in ['B', 'I', 'E', 'S']]

class XFUNConfig(datasets.BuilderConfig):
    """BuilderConfig for XFUN."""

    def __init__(self, lang, additional_langs=None, **kwargs):
        """
        Args:
            lang: string, language for the input text
            **kwargs: keyword arguments forwarded to super.
        """
        super(XFUNConfig, self).__init__(**kwargs)
        self.lang = lang
        self.additional_langs = additional_langs


class XFUN(datasets.GeneratorBasedBuilder):
    """XFUN dataset."""

    BUILDER_CONFIGS = [XFUNConfig(name=f"xfun.{lang}", lang=lang) for lang in _LANG]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "input_ids": datasets.Sequence(datasets.Value("int64")),
                    "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "labels": datasets.Sequence(
                        datasets.ClassLabel(
                            names=ner_labels
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.ClassLabel(names=labels),
                        }
                    ),

                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": [f"{_URL}{self.config.lang}.train.json", f"{_URL}{self.config.lang}.train.zip"],
            "val": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
            "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        train_files_for_many_langs = [downloaded_files["train"]]
        val_files_for_many_langs = [downloaded_files["val"]]
        test_files_for_many_langs = [downloaded_files["test"]]

        if self.config.additional_langs:
            additional_langs = self.config.additional_langs.split("+")
            if "all" in additional_langs:
                additional_langs = [lang for lang in _LANG if lang != self.config.lang]
            for lang in additional_langs:
                urls_to_download = {"train": [f"{_URL}{lang}.train.json", f"{_URL}{lang}.train.zip"]}
                additional_downloaded_files = dl_manager.download_and_extract(urls_to_download)
                train_files_for_many_langs.append(additional_downloaded_files["train"])

        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files_for_many_langs}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files_for_many_langs}
            ),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        print(filepaths)
        labels = set()

        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r") as f:
                data = json.load(f)

            for doc in data["documents"][:20]:
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(doc["img"]["fpath"])
                document = doc["document"]
                tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
                entities = []

                entity_id_to_index_map = {}
                empty_entity = set()
                for line in document:
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue

                    tokenized_inputs = self.tokenizer(
                        line["text"],
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                        return_attention_mask=False,
                    )
                    text_length = 0
                    ocr_length = 0
                    bbox = []
                    last_box = None
                    for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                        if token_id == 6:
                            bbox.append(None)
                            continue
                        text_length += offset[1] - offset[0]
                        tmp_box = []
                        while ocr_length < text_length:
                            ocr_word = line["words"].pop(0)
                            ocr_length += len(
                                self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                            )
                            tmp_box.append(simplify_bbox(ocr_word["box"]))
                        if len(tmp_box) == 0:
                            tmp_box = last_box
                        bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                        last_box = tmp_box
                    bbox = [
                        [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                        for i, b in enumerate(bbox)
                    ]
                    if line["label"] == "other":
                        label = ["O"] * len(bbox)
                    else:
                        if len(bbox) == 1:
                            label = [f"S-{line['label']}"]
                        else:
                            label = [f"I-{line['label']}"] * len(bbox)
                            if line['start_flag']:
                                label[0] = f"B-{line['label']}"
                            if line['end_flag']:
                                label[-1] = f"E-{line['label']}"

                    for l in label:
                        labels.add(l)

                    tokenized_inputs.update({"bbox": bbox, "labels": label})
                    if label[0] != "O":
                        # entity_id_to_index_map:每个实体对应一个唯一id，为每个id按照顺序重新索引
                        entity_id_to_index_map[line["id"]] = len(entities)
                        entities.append(
                            {
                                "start": len(tokenized_doc["input_ids"]),
                                "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                                "label": line["label"],
                            }
                        )
                    for i in tokenized_doc:
                        tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]

                chunk_size = 512
                for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                    item = {}
                    for k in tokenized_doc:
                        item[k] = tokenized_doc[k][index: index + chunk_size]
                    # entities_in_this_span保存切分后的所有实体
                    entities_in_this_span = []
                    # global_to_this_span 为切分的实体id-->切分后的实体id
                    global_to_local_map = {}
                    for entity_id, entity in enumerate(entities):
                        if (
                                index <= entity["start"] < index + chunk_size
                                and index <= entity["end"] < index + chunk_size
                        ):
                            entity["start"] = entity["start"] - index
                            entity["end"] = entity["end"] - index
                            global_to_local_map[entity_id] = len(entities_in_this_span)
                            entities_in_this_span.append(entity)

                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "entities": entities_in_this_span,

                        }
                    )
                    yield f"{doc['id']}_{chunk_id}", item


def generate_examples():
    filepaths = [['/work/Datasets/Doc-understanding/invoice_data/xfund-format/zh.test.json',
                  '/work/Datasets/Doc-understanding/invoice_data/xfund-format']]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    # labels = ['OTHER', 'QUESTION', 'ANSWER', 'KV']
    # label2id = {label: index for index, label in enumerate(labels)}
    # id2label = {index: label for index, label in enumerate(labels)}
    seq_labels = set()
    for filepath in filepaths:
        logger.info("Generating examples from = %s", filepath)
        with open(filepath[0], "r") as f:
            data = json.load(f)

        for doc in data["documents"]:
            doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
            print('--' * 50)
            print(doc['img']['fpath'])
            image, size = load_image(doc["img"]["fpath"])
            document = doc["document"]
            tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
            entities = []

            entity_id_to_index_map = {}
            empty_entity = set()
            for line in document:
                # if line['text'].startswith('Here is yet another version of the quest'):
                #     print(1 + 2)

                if len(line["text"]) == 0:
                    empty_entity.add(line["id"])
                    continue

                tokenized_inputs = tokenizer(
                    line["text"],
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    return_attention_mask=False,
                )
                text_length = 0
                ocr_length = 0
                bbox = []
                last_box = None

                for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                    if token_id == 6:
                        # None的坐标，使用下个bbox的[bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]]
                        bbox.append(None)
                        continue
                    text_length += offset[1] - offset[0]
                    # if offset[1] - offset[0] > 1:
                    #     print(33)
                    tmp_box = []
                    while ocr_length < text_length:  # tokenizer可能分词出 “时间”这样的token，而原始数据是每个字的bbox，所以需要合并每个字的bbox
                        ocr_word = line["words"].pop(0)
                        ocr_length += len(
                            tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                        )
                        tmp_box.append(simplify_bbox(ocr_word["box"]))
                    if len(tmp_box) == 0:
                        tmp_box = last_box
                    bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                    last_box = tmp_box
                bbox = [
                    [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                    for i, b in enumerate(bbox)
                ]
                if line["label"] == "other":
                    label = ["O"] * len(bbox)
                else:
                    if len(bbox) == 1:
                        label = [f"S-{line['label']}"]
                    else:
                        label = [f"I-{line['label']}"] * len(bbox)
                        if line['start_flag']:
                            label[0] = f"B-{line['label']}"
                        if line['end_flag']:
                            label[-1] = f"E-{line['label']}"

                for l in label:
                    seq_labels.add(l)
                tokenized_inputs.update({"bbox": bbox, "labels": label})
                if label[0] != "O":
                    entity_id_to_index_map[line["id"]] = len(entities)
                    entities.append(
                        {
                            "start": len(tokenized_doc["input_ids"]),
                            "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                            "label": line["label"],
                        }
                    )
                for i in tokenized_doc:
                    tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]

            chunk_size = 512  # 512
            overlap = 20
            for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
                item = {}
                for k in tokenized_doc:
                    item[k] = tokenized_doc[k][index: index + chunk_size]
                entities_in_this_span = []
                global_to_local_map = {}
                for entity_id, entity in enumerate(entities):
                    if (
                            index <= entity["start"] < index + chunk_size
                            and index <= entity["end"] < index + chunk_size
                    ):
                        entity["start"] = entity["start"] - index
                        entity["end"] = entity["end"] - index
                        global_to_local_map[entity_id] = len(entities_in_this_span)
                        entities_in_this_span.append(entity)

                item.update(
                    {
                        "id": f"{doc['id']}_{chunk_id}",
                        "image": image,
                        "entities": entities_in_this_span,
                    }
                )
                # print(item)
                # yield f"{doc['id']}_{chunk_id}", item
    print(seq_labels)


if __name__ == '__main__':
    generate_examples()
