from models.AbstractModel import *

if __name__ == '__main__':
    dataset = ConfigurableDataset(
        book_sets=[BookSet.HCM],
        lm_level=LanguageModelLevel.CHAR_LEVEL,
        seq_len=32,
        batch_size=8,
    )
    print(len(dataset.data))
    print(len(dataset))
    print(set([d[0].shape for d in dataset]))
