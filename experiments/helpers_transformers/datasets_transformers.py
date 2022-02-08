import os
import torch

from PIL import Image

def normalize_bbox(bbox, width, height):
    return [
         int(1000 * (bbox[0] / width)),
         int(1000 * (bbox[1] / height)),
         int(1000 * (bbox[2] / width)),
         int(1000 * (bbox[3] / height)),
     ]


def prepare_dataframe(df, metadata, crop=False):
    df['words'] = df.apply(lambda x: [x['text'][s:s+l] for l, s in zip(x['text_coords']['l'], x['text_coords']['s'])], axis=1)

    if not crop:
        df['bbox'] = df['text_coords'].apply(lambda x: list(zip(x['x'], x['y'], x['x'] + x['width'], x['y'] + x['height'])))

        def realign_bbox(bbox, row):
            bbox = (min(max(0, bbox[0] - row['x']), row['width']),
                    min(max(0, bbox[1] - row['y']), row['height']),
                    min(max(0, bbox[2] - row['x']), row['width']),
                    min(max(0, bbox[3] - row['y']), row['height']))
            return bbox

        df['bbox'] = df.apply(lambda row: [normalize_bbox(realign_bbox(bbox, row), row['width'], row['height']) for bbox in row['bbox']], axis=1)

    else:
        df['ratio'] = df['pid'].apply(lambda x: metadata[x]['resized_height']/metadata[x]['height'])
        df['x'] = df.apply(lambda row: int(row['ratio']*row['x']), axis=1)
        df['y'] = df.apply(lambda row: int(row['ratio']*row['y']), axis=1)
        df['width'] = df.apply(lambda row: int(row['ratio']*row['width']), axis=1)
        df['height'] = df.apply(lambda row: int(row['ratio']*row['height']), axis=1)
        df['bbox'] = df['text_coords'].apply(lambda x: list(zip(x['x'], x['y'], x['x'] + x['width'], x['y'] + x['height'])))

        def realign_bbox(bbox, row):
            bbox = (min(max(0, int(bbox[0]*row['ratio']) - row['x']), row['width']),
                    min(max(0, int(bbox[1]*row['ratio']) - row['y']), row['height']),
                    min(max(0, int(bbox[2]*row['ratio']) - row['x']), row['width']),
                    min(max(0, int(bbox[3]*row['ratio']) - row['y']), row['height']))
            return bbox

        df['bbox'] = df.apply(lambda row: [normalize_bbox(realign_bbox(bbox, row), row['width'], row['height']) for bbox in row['bbox']], axis=1)


class LayoutLMDataset(torch.utils.data.Dataset):

    def __init__(self, df, metadata, tokenizer, LE):
        max_seq_length = tokenizer.model_max_length
        pad_token_box = [0, 0, 0, 0]

        prepare_dataframe(df, metadata)

        self.encodings = []
        self.bbox = []
        for i, row in df.iterrows():
            assert len(row['words']) == len(row['bbox'])
            token_boxes = []
            for word, box in zip(row['words'], row['bbox']):
                word_tokens = tokenizer.tokenize(word)
                token_boxes.extend([box] * len(word_tokens))

            # Truncation of token_boxes
            special_tokens_count = 2
            if len(token_boxes) > max_seq_length - special_tokens_count:
                token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

            # add bounding boxes of cls + sep tokens
            token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

            encoding = tokenizer(' '.join(row['words']), padding='max_length', truncation=True)
            self.encodings.append(encoding)

            # Padding of token_boxes up the bounding boxes to the sequence length.
            input_ids = tokenizer(' '.join(row['words']), truncation=True)["input_ids"]
            padding_length = max_seq_length - len(input_ids)
            token_boxes += [pad_token_box] * padding_length

            self.bbox.append(token_boxes)

        self.labels = LE.transform(df['tag'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['bbox'] = torch.tensor(self.bbox[idx])
        return item

    def __len__(self):
        return len(self.labels)


class LayoutXLMDataset(torch.utils.data.Dataset):
    def __init__(self, df, metadata, processor, LE, images_path, ocr=False):
        prepare_dataframe(df, metadata, crop=True)
        self.encodings = []
        for i, row in df.iterrows():
            image = Image.open(os.path.join(images_path, row['pid'] + ".png")).convert("RGB")
            image = image.crop((row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height']))
            if ocr:
                encoding = processor(image, return_tensors="pt", padding="max_length", truncation=True, max_length=512)  # pt as in pytorch tensor
            else:
                encoding = processor(image, row['words'], boxes=row['bbox'], return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            self.encodings.append(encoding)

        self.labels = LE.transform(df['tag'])

    def __getitem__(self, idx):
        item = {key: val.squeeze() for key, val in self.encodings[idx].items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, LE):
        self.encodings = tokenizer(list(df['text']), padding="max_length", truncation=True)
        self.labels = LE.transform(df['tag'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
