import pandas as pd

filepath_dict = {'yelp':   '../data/sentiment_analysis/yelp_labelled.txt',
                     'amazon': '../data/sentiment_analysis/amazon_cells_labelled.txt',
                     'imdb':   '../data/sentiment_analysis/imdb_labelled.txt'}


def get_data(datasets=['yelp', 'amazon', 'imdb']):
    df_list = []
    for source, filepath in filepath_dict.items():
        if source in datasets:
            df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
            df['source'] = source  # Add another column filled with the source name
            df_list.append(df)
    df = pd.concat(df_list)
    return(df)
