TRAIN_SIZE = 100000
TEST_SIZE = 20000
PER_ROUND_TRAIN_SIZE = 500
PER_ROUND_TEST_SIZE = 280
IMAGE_DIM = 64
# DIST_NUM = 2
# EPOCH_NUM = 10


import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import multiprocessing as mp
mp.set_start_method("fork", force=True)  # Avoids issues on macOS

import argparse
parser = argparse.ArgumentParser(description='Generate datasets for ANDA')
parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
parser.add_argument('--scaling', type=int, default=0, help='Data scaler')
parser.add_argument('--epoch_num', type=int, default=5, help='Number of data changes during the training')
parser.add_argument('--n_clients', type=int, default=3, help='Number of clients')
args = parser.parse_args()
cur_seed = 42 + args.fold
np.random.seed(cur_seed)
torch.manual_seed(cur_seed)

if args.scaling == 0:
    DIST_NUM = 2
elif args.scaling == 1:
    DIST_NUM = 4
elif args.scaling == 2:
    DIST_NUM = 8
else:
    raise ValueError("Invalid scaling value. Must be 0, 1, or 2.")

EPOCH_NUM = args.epoch_num


# Download latest version
# path = kagglehub.dataset_download("ashery/chexpert")
# print("Path to dataset files:", path)

# kaggle saved
path = '/home/mohan/.cache/kagglehub/datasets/ashery/chexpert/versions/1'

# sampled saved
path_sampled = './data/saved_chex'
os.makedirs('./data/cur_datasets', exist_ok=True)
os.makedirs(path_sampled, exist_ok=True)

def generate_DA_dist(
    dist_bank: list,
    DA_epoch_locker_num: int,
    DA_max_dist: int,
    DA_continual_divergence: bool
) -> list:
    lst = []
    while len(lst) < DA_epoch_locker_num:
        # reaching DA_max_dist
        if len(set(lst)) == DA_max_dist:
            lst.append(lst[-1]) if DA_continual_divergence else lst.append(np.random.choice(lst))
        else:
            # update dist_bank 
            if len(lst) > 0 and DA_continual_divergence:
                dist_bank = [x for x in dist_bank if x not in lst or x == lst[-1]]
            lst.append(np.random.choice(dist_bank))
    
    return lst

def split_to_K_dist(
    # features: pd.DataFrame,
    # labels: pd.DataFrame,
    start_sample: int = 0,
    n_sample: int = 1000,
    image_dim: int = 64,
    dist_num: int = 8, #[2,4,8]
    data_path: str = '/home/mohan/.cache/kagglehub/datasets/ashery/chexpert/versions/1',
    save_dir: str = './data/saved_chex'
) -> list:
    '''
        output #dist_num dicts with np features and labels
    '''
    assert dist_num in [2, 4, 8], "dist_num must be in [2, 4, 8]"

    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters used for filename suffix
    suffix = f"{TRAIN_SIZE}_{TEST_SIZE}"
    img_file = os.path.join(save_dir, f"images_{suffix}.npy")
    label_file = os.path.join(save_dir, f"labels_{suffix}.npy")
    df_file = os.path.join(save_dir, f"df_{suffix}.pkl")

    # pre-load
    if os.path.exists(img_file) and os.path.exists(label_file) and os.path.exists(df_file):
        print("Loading saved image, label arrays, and dataframe...")
        all_images = np.load(img_file)
        all_labels = np.load(label_file)
        all_df = pd.read_pickle(df_file)

    # save new
    else:
        print("Preprocessing data from scratch and saving...")
        # load all data
        csv_path = os.path.join(data_path, "train.csv")
        all_df = pd.read_csv(csv_path)
        all_df.iloc[:, 5:] = all_df.iloc[:, 5:].replace(-1.0, 1.0).fillna(0.0)

        # Adjust image paths
        all_paths = [
            data_path + path.replace('CheXpert-v1.0-small', '')
            for path in all_df['Path']
        ]

        # Get the sample slice
        all_df = all_df.iloc[start_sample:n_sample, :]
        all_labels = all_df.iloc[:, 5:].to_numpy(dtype=np.float32)

        # Load and preprocess images
        def load_images(image_paths, n_sample=n_sample, image_dim=image_dim):
            images = []
            transform = transforms.Compose([
                transforms.Resize((image_dim, image_dim)),
                transforms.ToTensor()
            ])
            for path in tqdm(image_paths[start_sample:n_sample], desc="Loading images"):
                try:
                    img = Image.open(path).convert('L')
                    img = transform(img)
                    images.append(img.numpy())
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
            return np.stack(images)

        all_images = load_images(all_paths)

        # Save all three components
        np.save(img_file, all_images)
        np.save(label_file, all_labels)
        all_df.to_pickle(df_file)


    cur_data_list = []
    # split by Frontal/Lateral
    if dist_num == 2:
        frontal_mask = all_df['Frontal/Lateral'] == 'Frontal'
        lateral_mask = all_df['Frontal/Lateral'] == 'Lateral'

        # Split DataFrame
        df_frontal = all_df[frontal_mask].reset_index(drop=True)
        df_lateral = all_df[lateral_mask].reset_index(drop=True)

        # Split images and labels accordingly
        images_frontal = all_images[frontal_mask.to_numpy()]
        images_lateral = all_images[lateral_mask.to_numpy()]

        labels_frontal = all_labels[frontal_mask.to_numpy()]
        labels_lateral = all_labels[lateral_mask.to_numpy()]

        cur_data_list = [
            {"images": images_frontal, "labels": labels_frontal},
            {"images": images_lateral, "labels": labels_lateral}
        ]        

    # split by Frontal/Lateral and age over 60 or not
    elif dist_num == 4:
        # Create masks for each condition
        frontal_mask = all_df['Frontal/Lateral'] == 'Frontal'
        lateral_mask = all_df['Frontal/Lateral'] == 'Lateral'

        age_mask_60 = all_df['Age'] >= 60
        age_mask_below_60 = all_df['Age'] < 60

        # Combine masks to get 4 groups
        mask_frontal_young = frontal_mask & age_mask_below_60
        mask_frontal_old = frontal_mask & age_mask_60
        mask_lateral_young = lateral_mask & age_mask_below_60
        mask_lateral_old = lateral_mask & age_mask_60

        cur_data_list = []
        for mask in [mask_frontal_young, mask_frontal_old, mask_lateral_young, mask_lateral_old]:
            mask_np = mask.to_numpy()
            cur_data_list.append({
                "images": all_images[mask_np],
                "labels": all_labels[mask_np]
            })

    # split by Frontal/Lateral, age over 60 or not, and gender
    elif dist_num == 8:

        # Define masks
        frontal_mask = all_df['Frontal/Lateral'] == 'Frontal'
        lateral_mask = all_df['Frontal/Lateral'] == 'Lateral'

        age_mask_60 = all_df['Age'] >= 60
        age_mask_below_60 = all_df['Age'] < 60

        female_mask = all_df['Sex'] == 'Female'
        male_mask = all_df['Sex'] == 'Male'

        # Prepare all 8 combinations
        conditions = [
            frontal_mask & age_mask_below_60 & female_mask,
            frontal_mask & age_mask_below_60 & male_mask,
            frontal_mask & age_mask_60 & female_mask,
            frontal_mask & age_mask_60 & male_mask,
            lateral_mask & age_mask_below_60 & female_mask,
            lateral_mask & age_mask_below_60 & male_mask,
            lateral_mask & age_mask_60 & female_mask,
            lateral_mask & age_mask_60 & male_mask,
        ]

        cur_data_list = []
        for mask in conditions:
            mask_np = mask.to_numpy()
            cur_data_list.append({
                "images": all_images[mask_np],
                "labels": all_labels[mask_np]
            })

    else:
        raise ValueError("dist_num must be in [2, 4, 8]")

    for i, data in enumerate(cur_data_list):
        print(f"Data {i}:")
        print(f"  Images shape: {data['images'].shape}")
        print(f"  Labels shape: {data['labels'].shape}")

    return cur_data_list



train_data_list = split_to_K_dist(start_sample=0,n_sample=TRAIN_SIZE, image_dim=IMAGE_DIM, dist_num=DIST_NUM, data_path=path)
test_data_list = split_to_K_dist(start_sample=TRAIN_SIZE,n_sample=TRAIN_SIZE+TEST_SIZE, image_dim=IMAGE_DIM, dist_num=DIST_NUM, data_path=path)


train_dist_list = []
last_dist_set = set()  # Use a set to avoid duplicates
last_dist_list = []
test_dist_list = []
dist_bank = list(range(1, DIST_NUM + 1))

for i in range(args.n_clients):
    cur_DA_dist = generate_DA_dist(dist_bank, DA_epoch_locker_num = EPOCH_NUM, DA_max_dist = 100, DA_continual_divergence = False)
    train_dist_list.append(cur_DA_dist)
    last_dist_set.add(cur_DA_dist[-1]) 
    last_dist_list.append(cur_DA_dist[-1])

print("Last dist set: ", last_dist_set)
print("Last dist list: ", last_dist_list)

for i in range(args.n_clients):
    test_dist_list.append(np.random.choice(last_dist_list))

anda_dataset = []

for client_Count in range(args.n_clients):
    print(f"Client: {client_Count}")

    # generate drifting
    train_dist = train_dist_list[client_Count]
    test_dist = test_dist_list[client_Count]
    
    lockers = torch.linspace(0, 1, steps=EPOCH_NUM + 1)[:-1].tolist()

    print("Train distribution: ", train_dist,
            "\nTest distribution: ", test_dist,
            "\nEpoch lockers: ", lockers,
            "\n")

    # training subsets

    # Loop through feature chunks and label chunks
    for i in range(EPOCH_NUM):
        # Get angle and color from the pattern bank based on the train_dist
        cur_data_dict = train_data_list[train_dist[i] - 1]

        train_image_size = min(PER_ROUND_TRAIN_SIZE, cur_data_dict['labels'].shape[0])

        # load the data of this dict first
        cur_images = cur_data_dict['images']
        cur_labels = cur_data_dict['labels']

        # permute
        permuted_indices = torch.randperm(cur_images.shape[0])
        cur_images = cur_images[permuted_indices]
        cur_labels = cur_labels[permuted_indices]

        # sample
        cur_images = cur_data_dict['images'][:train_image_size]
        cur_labels = cur_data_dict['labels'][:train_image_size]


        if i == 0:
            # print size of feature_chunk and label_chunk
            print("Feature chunk size: ", cur_images.shape)
            print("Label chunk size: ", cur_labels.shape)

        # Append the cumulative data to rearranged_data
        anda_dataset.append({
            'train': True,
            'features': cur_images,
            'labels': cur_labels,
            'client_number': client_Count,
            'epoch_locker_indicator': lockers[i],
            'epoch_locker_order': i,
            'cluster': train_dist[i],
            'train_dist': train_dist,
            'test_dist': test_dist
        })

    # testing set
    cur_test_data_dict = test_data_list[test_dist - 1]

    test_image_size = min(PER_ROUND_TEST_SIZE, cur_test_data_dict['labels'].shape[0])

    # load the data of this dict first
    cur_images = cur_test_data_dict['images']
    cur_labels = cur_test_data_dict['labels']

    # permute
    permuted_indices = torch.randperm(cur_images.shape[0])
    cur_images = cur_images[permuted_indices]
    cur_labels = cur_labels[permuted_indices]

    # sample
    cur_images = cur_data_dict['images'][:test_image_size]
    cur_labels = cur_data_dict['labels'][:test_image_size]


    anda_dataset.append({
        'train': False,
        'features': cur_images,
        'labels': cur_labels,
        'client_number': client_Count,
        'epoch_locker_indicator': -1.0,
        'epoch_locker_order': -1,
        'cluster': test_dist,
        'ideal_dist_num': len(last_dist_set),
        'train_dist': train_dist,
        'test_dist': test_dist
    })




# complex format as training drifting
drifting_log = {}
client_distribution = {}
for dataset in anda_dataset:
    client_number = dataset['client_number']
    cur_drifting_round = int(20 * dataset['epoch_locker_indicator']) if dataset['epoch_locker_indicator'] != -1 else -1

    # save data file      
    filename = f'./data/cur_datasets/client_{client_number}_round_{cur_drifting_round}.npy'
    np.save(filename, dataset)
    print(f"Data for client {client_number} round {cur_drifting_round} saved")
    
    # save client distribution during training
    client_distribution[client_number] = dataset['train_dist']

    # log drifting round info
    if client_number not in drifting_log:
        drifting_log[client_number] = []
    drifting_log[client_number].append(cur_drifting_round)

# print(", ".join(f"{key}: {value}" for key, value in drifting_log.items()))

# save log file
np.save(f'./data/cur_datasets/drifting_log.npy', drifting_log)
np.save(f'./data/cur_datasets/client_distribution.npy', client_distribution)




print("Datasets saved successfully!")


