from utils import *
from models.Model import Baseline, ModelAttention, ModelSAGE, ModelGATConv, ModelAttentiveFP, ModelGATPerso, ModelGATwMLP, ModelTransformer , ModelGPS, ModelSuperGAT, ModelVGAE, ModelGINE, ModelTransformerv2
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
from models.VotingClassifier import VotingClassifier
from models.graph_encoder import *



# Setup save directories
MODEL_SURNAME =  'SoftVoting'
SUBMISSION_DIR = os.path.join('../submissions/', MODEL_SURNAME, '')
SAVE_DIR = os.path.join('../saves', MODEL_SURNAME, '')

if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    



# Reproducibility
seed = 3407
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)


# Setup the text encoder
model_name1 = 'distilbert-base-uncased'
model_name= 'distilbert-base-uncased-finetuned-sst-2-english'
#model_name = 'BAAI/bge-reranker-large'
#model_name = 'BAAI/llm-embedder'
#model_name = 'facebook/fasttext-language-identification'
#model_name = 'facebook/bart-large-cnn'
#model_name = 'lucadiliello/bart-small'
#model_name = 'bheshaj/bart-large-cnn-small-xsum-5epochs'
#model_name = 'google/pegasus-large'
#model_name = 'recobo/chemical-bert-uncased'
#model_name = 'allenai/scibert_scivocab_uncased'
#model_name = 'gokceuludogan/ChemBERTaLM'
#model_name = 'seyonec/PubChem10M_SMILES_BPE_450k'
#model_name = 'alvaroalon2/biobert_chemical_ner'
#model_name = 'jonas-luehrs/distilbert-base-uncased-MLM-scirepeval_fos_chemistry'
#model_name = 'nlpie/tiny-biobert'
#model_name = 'dmis-lab/biobert-v1.1'
#model_name = 'microsoft/deberta-v3-base'
#model_name = 'DeepChem/ChemBERTa-10M-MLM'
#model_name = 'DeepChem/SmilesTokenizer_PubChem_1M'
#model_name = 'unikei/bert-base-smiles'
#model_name = 'matr1xx/scibert_scivocab_uncased-finetuned-mol'
#model_name = 'ghadeermobasher/BC5CDR-Chemical-Disease-balanced-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
#model_name = 'FelixChao/vicuna-7B-chemical'
#model_name = 'JuIm/SMILES_BERT'
#model_name = 'nlpie/bio-tinybert'
#model_name = 'yashpatil/biobert-tiny-model'
#model = 'albert/albert-base-v2'

tokenizer = AutoTokenizer.from_pretrained(model_name)


gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]


# Train on GPU if possible (it is actually almost mandatory considering the size of the model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if device != torch.device("cpu"):
    print('================ GPU FOUND ================')
    print('Using device: {}'.format(device))
    print('GPU found: {}'.format(torch.cuda.get_device_name(0)))
    print('GPU memory: {:.3f} MB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024))
else:
    print('================ NO GPU ================')

# Training hyperparameters
batch_size = 128


models = []
models.append(ModelGATwMLP(model_name=model_name1, nout=768, nhid=1024, n_heads=8, n_in=300, dropout=0.6))
models.append(ModelSuperGAT(model_name=model_name1, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.4))
            

    



# Submission

test_cids_dataset = GraphDataset(root='../data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='../data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()
test_loader = TorchGeoDataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)
test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
CosineSimilarity = None
SigmoidSimilarity = None

for (model,k) in enumerate(models):
    model_name_path = 'model_' + str(k) + '.pt'
    save_path = os.path.join(SAVE_DIR, model_name_path)
    model.to(device)
    model.eval()
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()
    
    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())
    CosineSimilarity = CosineSimilarity + cosine_similarity(text_embeddings, graph_embeddings)
    SigmoidSimilarity = SigmoidSimilarity + sigmoid_kernel(text_embeddings, graph_embeddings)

CosineSimilarity = CosineSimilarity/len(models)
solution = pd.DataFrame(CosineSimilarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(os.path.join(SUBMISSION_DIR, 'COSINEsubmission.csv'), index=False)
print('submission saved to: {}'.format(os.path.join(SUBMISSION_DIR, 'COSINEsubmission.csv')))
print('================ DONE ================')

SigmoidSimilarity = SigmoidSimilarity/len(models)
solution = pd.DataFrame(SigmoidSimilarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(os.path.join(SUBMISSION_DIR, 'Sigmoidsubmission.csv'), index=False)
print('submission saved to: {}'.format(os.path.join(SUBMISSION_DIR, 'Sigmoidsubmission.csv')))
print('================ DONE ================')


