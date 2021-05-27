import pandas as pd

folder_path = "D:\\deep_learning_experiments"
lankadeepa_data_path = folder_path + "\\sinhala_data\\lankadeepa_tagged_comments.csv"
gossip_lanka_data_path = folder_path + "\\sinhala_data\\gossip_lanka_tagged_comments.csv"

lankadeepa_data = pd.read_csv(lankadeepa_data_path)[:9059]
gossipLanka_data = pd.read_csv(gossip_lanka_data_path)
gossipLanka_data = gossipLanka_data.drop(columns=['Unnamed: 3'])

word_embedding_path = folder_path

all_data = pd.concat([lankadeepa_data,gossipLanka_data], ignore_index=True)
print(all_data)