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
# ner_labels = ['O'] + [pre + '-' + label for label in labels for pre in ['B', 'I', 'E', 'S']]
ner_labels = {'B-company', 'E-name', 'I-total', 'B-date', 'B-price', 'E-cnt', 'E-date', 'S-total', 'I-company',
              'B-total', 'I-date', 'I-name', 'S-cnt', 'B-name', 'E-total', 'I-price', 'S-price', 'E-company', 'E-price',
              'O', 'I-cnt', 'B-cnt'}


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
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                            "start_index": datasets.Value("int64"),
                            "end_index": datasets.Value("int64"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "test": [f"{_URL}{self.config.lang}.test.json", f"{_URL}{self.config.lang}.test.zip"],
            # "test": [f"{_URL}{self.config.lang}.val.json", f"{_URL}{self.config.lang}.val.align.zip"],
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        test_files_for_many_langs = [downloaded_files["test"]]

        logger.info(f"Training on {self.config.lang} with additional langs({self.config.additional_langs})")
        logger.info(f"Evaluating on {self.config.lang}")
        logger.info(f"Testing on {self.config.lang}")
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        print(filepaths)

        for filepath in filepaths:
            logger.info("Generating examples from = %s", filepath)
            with open(filepath[0], "r") as f:
                data = json.load(f)

            for doc in data["documents"]:
                doc["img"]["fpath"] = os.path.join(filepath[1], doc["img"]["fname"])
                image, size = load_image(doc["img"]["fpath"])
                document = doc["document"]
                tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
                entities = []
                relations = []
                id2label = {}
                entity_id_to_index_map = {}
                empty_entity = set()
                for line in document:
                    if len(line["text"]) == 0:
                        empty_entity.add(line["id"])
                        continue
                    id2label[line["id"]] = line["label"]
                    relations.extend([tuple(sorted(l)) for l in line["linking"]])
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
                        label = [f"I-{line['label']}"] * len(bbox)
                        label[0] = f"B-{line['label']}"
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
                relations = list(set(relations))
                relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
                kvrelations = []
                for rel in relations:
                    pair = [id2label[rel[0]], id2label[rel[1]]]
                    if pair == ["question", "answer"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                        )
                    elif pair == ["answer", "question"]:
                        kvrelations.append(
                            {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                        )
                    else:
                        continue

                def get_relation_span(rel):
                    bound = []
                    for entity_index in [rel["head"], rel["tail"]]:
                        bound.append(entities[entity_index]["start"])
                        bound.append(entities[entity_index]["end"])
                    return min(bound), max(bound)

                relations = sorted(
                    [
                        {
                            "head": rel["head"],
                            "tail": rel["tail"],
                            "start_index": get_relation_span(rel)[0],
                            "end_index": get_relation_span(rel)[1],
                        }
                        for rel in kvrelations
                    ],
                    key=lambda x: x["head"],
                )
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
                    relations_in_this_span = []
                    for relation in relations:
                        # relation_span: start_index前实体的start，end_index尾实体的end
                        if (
                                index <= relation["start_index"] < index + chunk_size
                                and index <= relation["end_index"] < index + chunk_size
                        ):
                            relations_in_this_span.append(
                                {
                                    "head": global_to_local_map[relation["head"]],  # 切分后的头实体的索引
                                    "tail": global_to_local_map[relation["tail"]],  # 切分后的尾实体的索引
                                    "start_index": relation["start_index"] - index,
                                    "end_index": relation["end_index"] - index,
                                }
                            )
                    item.update(
                        {
                            "id": f"{doc['id']}_{chunk_id}",
                            "image": image,
                            "entities": entities_in_this_span,
                            "relations": relations_in_this_span,
                        }
                    )
                    yield f"{doc['id']}_{chunk_id}", item


def generate_examples():
    filepaths = [['/work/Codes/layoutlmft/examples/XFUND-DATA-Gartner/zh.val.json',
                  '/work/Codes/layoutlmft/examples/XFUND-DATA-Gartner/zh.val']]

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    # labels = ['OTHER', 'QUESTION', 'ANSWER', 'KV']
    # label2id = {label: index for index, label in enumerate(labels)}
    # id2label = {index: label for index, label in enumerate(labels)}

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
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()
            for line in document:
                # if line['text'].startswith('Here is yet another version of the quest'):
                #     print(1 + 2)

                if len(line["text"]) == 0:
                    empty_entity.add(line["id"])
                    continue
                id2label[line["id"]] = line["label"]
                relations.extend([tuple(sorted(l)) for l in line["linking"]])
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
                print(line['text'])
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
                    label = [f"I-{line['label']}"] * len(bbox)
                    label[0] = f"B-{line['label']}"
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
            relations = list(set(relations))
            relations = [rel for rel in relations if rel[0] not in empty_entity and rel[1] not in empty_entity]
            kvrelations = []
            for rel in relations:
                pair = [id2label[rel[0]], id2label[rel[1]]]
                if pair == ["question", "answer"]:
                    kvrelations.append(
                        {"head": entity_id_to_index_map[rel[0]], "tail": entity_id_to_index_map[rel[1]]}
                    )
                elif pair == ["answer", "question"]:
                    kvrelations.append(
                        {"head": entity_id_to_index_map[rel[1]], "tail": entity_id_to_index_map[rel[0]]}
                    )
                else:
                    continue

            def get_relation_span(rel):
                bound = []  # 获取两个实体间的最大距离的位置坐标
                for entity_index in [rel["head"], rel["tail"]]:
                    bound.append(entities[entity_index]["start"])
                    bound.append(entities[entity_index]["end"])
                return min(bound), max(bound)

            relations = sorted(
                [
                    {
                        "head": rel["head"],
                        "tail": rel["tail"],
                        "start_index": get_relation_span(rel)[0],
                        "end_index": get_relation_span(rel)[1],
                    }
                    for rel in kvrelations
                ],
                key=lambda x: x["head"],
            )
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
                relations_in_this_span = []

                for relation in relations:
                    if (
                            index <= relation["start_index"] < index + chunk_size
                            and index <= relation["end_index"] < index + chunk_size
                    ):
                        relations_in_this_span.append(
                            {
                                "head": global_to_local_map[relation["head"]],
                                "tail": global_to_local_map[relation["tail"]],
                                "start_index": relation["start_index"] - index,
                                "end_index": relation["end_index"] - index,
                            }
                        )
                item.update(
                    {
                        "id": f"{doc['id']}_{chunk_id}",
                        "image": image,
                        "entities": entities_in_this_span,
                        "relations": relations_in_this_span,
                    }
                )
                # print(item)
                # yield f"{doc['id']}_{chunk_id}", item


if __name__ == '__main__':
    generate_examples()
