import numpy as np
from tqdm import tqdm
import torch
from gtda.graphs import GraphGeodesicDistance
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, FlagserPersistence
import numpy as np
import warnings
from scipy.sparse import coo_matrix

def weights_to_coo(weights_matrix):
    """
    Convert a matrix of weights to COO format.

    Parameters:
    - weights_matrix (numpy.ndarray): Matrix of weights.

    Returns:
    - coo_matrix: COO format sparse matrix.
    """
    # Get non-zero indices and values from the weights_matrix
    row_indices, col_indices = np.nonzero(weights_matrix)
    values = weights_matrix[row_indices, col_indices]
    # Create COO format matrix
    coo_mat = coo_matrix((values, (row_indices, col_indices)), shape=weights_matrix.shape)
    return coo_mat


def histogram_entropy(hist):
    total = np.sum(hist)
    if total == 0:
        return 0
    probabilities = hist / total
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small value to avoid log(0)
    return entropy


def compute_features(diagrams):
    diagrams_list = diagrams[0]
    number_of_bars_h0 = 0
    number_of_bars_h1 = 0
    length_h0 = list()
    length_h1 = list()
    diag_h0 = list()
    diag_h1 = list()
    for bar in diagrams_list:
        birth = bar[0]
        death = bar[1]
        homology_type = bar[2]
        if homology_type == 0. :
            diag_h0.append(bar)
            length = death - birth
            length_h0.append(length)
            number_of_bars_h0 +=1
        if homology_type == 1. :
            diag_h1.append(bar)
            length = death - birth
            length_h1.append(length)
            number_of_bars_h1 +=1


    if number_of_bars_h0 == 0 :
        entr_h0 = 0
        time_of_birth_longest_h0= 0
        time_of_death_longest_h0= 0
        sum_of_lenghts_h0 = 0
        mean_of_lenghts_h0= 0
        var_of_lenghts_h0 = 0
    else:
        entr_h0 = histogram_entropy(diag_h0)
        time_of_birth_longest_h0 = diag_h0[length_h0.index(max(length_h0))][0]
        time_of_death_longest_h0 = diag_h0[length_h0.index(max(length_h0))][1]
        sum_of_lenghts_h0 = sum(length_h0)
        mean_of_lenghts_h0 = np.mean(np.array(length_h0))
        var_of_lenghts_h0 = np.var(np.array(length_h0))
        
    if number_of_bars_h1 == 0 :
        entr_h1 = 0
        time_of_birth_longest_h1= 0
        time_of_death_longest_h1= 0
        sum_of_lenghts_h1 = 0
        mean_of_lenghts_h1= 0
        var_of_lenghts_h1 = 0
    else:
        entr_h1 = histogram_entropy(diag_h1)
        time_of_birth_longest_h1 = diag_h1[length_h1.index(max(length_h1))][0]
        time_of_death_longest_h1 = diag_h1[length_h1.index(max(length_h1))][1]
        sum_of_lenghts_h1 = sum(length_h1)
        mean_of_lenghts_h1 = np.mean(np.array(length_h1))
        var_of_lenghts_h1 = np.var(np.array(length_h1))

    feature_list = [number_of_bars_h0,        number_of_bars_h1,
                    time_of_birth_longest_h0, time_of_birth_longest_h1, 
                    time_of_death_longest_h0, time_of_death_longest_h1, 
                    sum_of_lenghts_h0,        sum_of_lenghts_h1, 
                    mean_of_lenghts_h0,       mean_of_lenghts_h1,
                    var_of_lenghts_h0,        var_of_lenghts_h1,
                    entr_h0,                  entr_h1]
    return feature_list
      

def undirected_features(model, train_loader, device):
    print("Undirected features calculations is starting")
    all_features = list()
    for data, labels in tqdm(train_loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(data)

        attention_matrices = outputs.attentions  #Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
        attention_map = attention_matrices[-1] # from the last layer
        for i in range(attention_map.shape[0]):
            list_features = list()
            attention_for_sample = attention_map[i]
            attention = attention_for_sample[-1, :, :].cpu().numpy()
            X = [attention]
            VR = VietorisRipsPersistence(metric="precomputed")
            diagrams = VR.fit_transform(X)
            features = compute_features(diagrams)
            all_features.append(np.array(features))
    print("Undirected features calculations are calculated") 
    return all_features


def directed_features(model, train_loader, device):
    print("Directed features calculations is starting")
    all_features = list()
    for data, labels in tqdm(train_loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(data)

        attention_matrices = outputs.attentions  #Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
        attention_map = attention_matrices[-1] # from the last layer
        for i in range(attention_map.shape[0]):
            list_features = list()
            attention_for_sample = attention_map[i]
            attention = attention_for_sample[-1, :, :].cpu().numpy()
            np.fill_diagonal(attention, 0)
            a = FlagserPersistence().fit_transform([weights_to_coo(attention)])
            a[a == np.inf] = 1000
            features = compute_features(a)
            all_features.append(np.array(features))
    print("Directed features calculations are calculated") 
    return all_features