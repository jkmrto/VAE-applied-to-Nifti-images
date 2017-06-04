import tensorflow as tf
from sklearn import svm
from sklearn.metrics import confusion_matrix
from lib.data_loader import mri_atlas
import settings
from lib.vae import VAE
from lib.data_loader import MRI_stack_NORAD
from sklearn.model_selection import train_test_split
from sklearn import metrics

ARCHITECTURE = [28**2, # 784 pixels
                500, 500, # intermediate encoding
                10]# latent space dims
                # 50]
# (and symmetrically back out again)

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 5E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}


def normalize_array(array):
    out = array / array.max()
    return out, array.max

region_selected = 8
region_voxels_index = mri_atlas.load_atlas_mri()[region_selected]
dict_norad = MRI_stack_NORAD.get_gm_stack()  # 'stack' 'voxel_index' 'labels'


region_voxels_values = dict_norad['stack'][:, region_voxels_index]
region_voxels_values, max_denormalize = normalize_array(region_voxels_values)
region_voxels_label = dict_norad['labels']

X_train, X_test, y_train, y_test = train_test_split(region_voxels_values, region_voxels_label,
                                                    test_size=0.2, random_state=0)

path_save = settings.path_to_project + "/meta/region_8_170404_2005_vae_11832_1500_1000_500_200-2000"
sess = tf.Session()
new_saver = tf.train.import_meta_graph(path_save + ".meta")
new_saver.restore(sess, path_save)

# Hyperparameters and architecture is not used in loading setup
v = VAE.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=settings.LOG_DIR, meta_graph=path_save)


#Extracting main charactheristics -> 2 variables per sample
print("Coding training data")
code_train = v.encode(X_train)  # [mu, sigma]
print("Coding test data")
code_test = v.encode(X_test)  # [mu, sigma]

#code_sample = sample_gaussian(code[0], code[1])
#clf = svm.SVC('ovr')
#clf.fit(code[0], mnist_test_labels)
#X = [[0], [1], [2], [3]]
#Y = [0, 1, 2, 3]

# Fitting SVM
print("Training SVM")
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(code_train[0], y_train)    # Solo usando las media pq el codificador devuelve media y desviacino

# Testing time
print("Evaluating test samples")
dec = clf.decision_function(code_test[0])

print(metrics.roc_auc_score(y_true=y_test, y_score=dec))
#confusion_matrix(mnist_test_labels, dec.argmax(axis=1))


