# Generate synthetic AA sequences, use HDC to encode/classify, try to reveal
# the pattern using similarities.

from hyperdim_proteins import embed_sequences, py_generate_trimer_hdvs
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np


AMINO_ACIDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ."
KEY_TRIMER = "ADD"
N_DIM = 10_000
SEED = 1701


def bundle(hdv_list):
    out = np.sign(np.sum(np.stack(hdv_list), axis=0))
    out[out == 0] = 1
    return out


def cosine_sim_batch(v1, v2):
    v2_transpose = v2.T
    norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True)
    norm_v2 = np.linalg.norm(v2_transpose, axis=0, keepdims=True)
    normed = np.dot(norm_v1, norm_v2)
    return (np.dot(v1, v2_transpose) / normed).astype(np.float16)


# Generate hypervector info
trimer_hdvs = py_generate_trimer_hdvs(AMINO_ACIDS, SEED)
embed = lambda seqs: np.stack(embed_sequences(seqs, trimer_hdvs))

# Generate the protein sequences
sequences = [
    "".join(np.random.choice(list(AMINO_ACIDS), 200, replace=True))
    for _ in range(10000)
]
labels = np.zeros(len(sequences))
for ind, aa_seq in enumerate(sequences):
    if np.random.random() < 0.5:
        start_pos = int(np.random.uniform(0, 200 - 30))
        sequences[ind] = aa_seq[:start_pos] + KEY_TRIMER * 3 + aa_seq[(start_pos + 9) :]
        labels[ind] = 1
# Convert sequences to embeddings
sequence_embeds = embed(sequences)
# Split into train/test
train_embed, test_embed, train_lab, test_lab = train_test_split(
    sequence_embeds, labels, test_size=0.2
)

# Train: Bundle training data into prototypes
all_prototypes = np.zeros(shape=(2, N_DIM))
all_prototypes[0, :] = bundle(train_embed[train_lab == 0, :])
all_prototypes[1, :] = bundle(train_embed[train_lab == 1, :])

# Predict on unseen sequences
similarity_array = cosine_sim_batch(all_prototypes, test_embed)
pred_cat = np.argmax(similarity_array, axis=0)
accuracy = balanced_accuracy_score(test_lab, pred_cat)
print(f"pred_acc = {accuracy:.2%}")
sns.heatmap(confusion_matrix(test_lab, pred_cat, normalize="true"))
plt.show()

# Compare to trimers to find relationship
trimers, hdv_list = zip(*trimer_hdvs.items())
trimer_similarities = cosine_sim_batch(all_prototypes, np.stack(hdv_list))
print(np.array(trimers)[np.where(np.abs(trimer_similarities[1, :]) > 0.2)[0]])

# addaddadd = bundle([trimer_hdvs["ADD"], trimer_hdvs["ADD"], trimer_hdvs["ADD"]])
# sim = cosine_sim_batch(all_prototypes, addaddadd)
