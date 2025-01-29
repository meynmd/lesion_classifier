from pathlib import Path
import datetime
import argparse
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split


def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--data-root',
        '-d',
        type=Path,
        required=True   
    )
    arg_parser.add_argument(
        '--output-root', 
        '-o',
        type=Path,
        required=True
    )
    arg_parser.add_argument(
        '--do-split', 
        action='store_true'
    )

    return arg_parser.parse_args()


def enumerate_dataset(
    root_dir: Path,
    positive_label: str = 'Malignant',
    negative_label: str = 'Benign',
    accepted_exts: List[str] = ['jpg', 'JPG', 'png', 'PNG']
):
    pos_ex_dir = root_dir / positive_label
    neg_ex_dir = root_dir / negative_label

    data_list = []
    for label, subdir in [(1, pos_ex_dir), (0, neg_ex_dir)]:
        for ext in accepted_exts:
            for filename in subdir.glob(f'*.{ext}'):
                data_list.append({
                    'filename': filename, 
                    'label': label
                })
    
    return pd.DataFrame(data_list)


def main():
    # get console args
    args = get_args()
    data_root = args.data_root
    output_root = args.output_root

    if not output_root.exists():
        output_root.mkdir()

    df = enumerate_dataset(data_root)

    if args.do_split:
        idx_train, idx_val = train_test_split(
            df.index,
            test_size=0.05, 
            random_state=1
        )
        df.loc[idx_train, 'split'] = 'train'
        df.loc[idx_val, 'split'] = 'validation'
        df_train = df[df['split'] == 'train']
        df_val = df[df['split'] == 'validation']
        train_filename = output_root / 'train.csv'
        val_filename = output_root / 'validation.csv'
        print(f'saving {train_filename}, length {len(df_train)}')
        print(f'saving {val_filename}, length {len(df_val)}')
        df_train.to_csv(train_filename)
        df_val.to_csv(val_filename, index=False)
    else:
        df.loc[:, 'split'] = 'test'
        test_filename = output_root / 'test.csv'
        print(f'saving {test_filename}, length {len(df)}')
        df.to_csv(test_filename, index=False)


if __name__ == '__main__':
    main()