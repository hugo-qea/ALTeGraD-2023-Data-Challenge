from utils import *
from models.Model import Baseline, ModelAttention, ModelSAGE, ModelGATConv, ModelAttentiveFP, ModelGATPerso, ModelGATwMLP, ModelTransformer , ModelGPS, ModelSuperGAT, ModelVGAE, ModelGINE, ModelTransformerv2
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold





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
#model_name = 'distilbert-base-uncased'
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
val_dataset = GraphTextDataset(root='../data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='../data/', gt=gt, split='train', tokenizer=tokenizer)

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
nb_epochs = 8
batch_size = 32
learning_rate = 5e-5

# Setup the batch loaders
val_loader = TorchGeoDataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = TorchGeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#model = Baseline(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
#model = ModelAttention(model_name=model_name, n_in=300, nout=768, nhid=1024, attention_hidden=1024, dropout=0.6) # nout = bert model hidden dim
#model = ModelSAGE(model_name=model_name, n_in=300, nout=768, nhid=1000, sage_hidden=1000, dropout=0.3) # nout = bert model hidden dim
#model = ModelGATConv(model_name=model_name, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.3) # nout = bert model hidden dim
#model = ModelAttentiveFP(model_name=model_name, n_in=300, nout=768, nhid=1000, attention_hidden=1000, dropout=0.3) # nout = bert model hidden dim
#model = ModelGATPerso(model_name=model_name, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.6) # nout = bert model hidden dim
#model = ModelGATwMLP(model_name=model_name, n_in=300, nout=768, nhid=2048, n_heads=4, dropout=0.6) # nout = bert model hidden dim
#model = ModelTransformer(model_name=model_name, n_in=300, nout=768, nhid=764, n_heads=4, dropout=0.6) # nout = bert model hidden dim
#model = ModelGPS(model_name=model_name, n_in=300, nout=768, nhid=1024, n_heads=6, dropout=0.6) # nout = bert model hidden dim
#model = ModelSuperGAT(model_name=model_name, n_in=300, nout=768, nhid=768, n_heads=4, dropout=0.6) # nout = bert model hidden dim
#model = ModelVGAE(model_name=model_name, n_in=300, nout=768, nhid=300, n_heads=8, dropout=0.6) # nout = bert model hidden dim
#model = ModelGINE(model_name=model_name, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.6) # nout = bert model hidden dim
model = ModelTransformerv2(model_name=model_name, n_in=300, nout=300, nhid=100, n_heads=2, dropout=0.75) # nout = bert model hidden dim



model.to(device)

MODEL_SURNAME =  model.get_model_surname() + model_name +'n_heads=2,nhid=100,dropout=0.6' +'linear'
#MODEL_SURNAME = 'BASELINE_CLASSIC_BS'
SUBMISSION_DIR = os.path.join('../submissions/', MODEL_SURNAME, '')
SAVE_DIR = os.path.join('../saves', MODEL_SURNAME, '')
COMMENT = MODEL_SURNAME+'-lr'+str(learning_rate)+'-batch_size'+str(batch_size)

if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    

# Print summary of the graph encoder model
encoder = model.get_graph_encoder()
x = torch.randn(2000,300).to(device)
edge_index = torch.randint(32, size=(2, 2000)).to(device)
data = Data(x=x, edge_index=edge_index)
summary_graph_encoder = summary(encoder, data)
print(summary_graph_encoder)
f = open(os.path.join(SAVE_DIR, 'summary_graph_encoder.txt'), 'w')
f.write(summary_graph_encoder)
f.close()


# Number of folds for cross-validation
num_folds = 2  # Adjust as needed

# Initialize k-fold cross-validator
kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)



# Print summary of the text encoder model
# Generate random input data for the summary
input_ids = torch.randint(1000, size=(batch_size, 128)).to(device)
attention_mask = torch.randint(2, size=(batch_size, 128)).to(device)
summary_text_encoder = summary(model.get_text_encoder(), input_ids, attention_mask)
print(summary_text_encoder)
f = open(os.path.join(SAVE_DIR, 'summary_text_encoder.txt'), 'w')
f.write(summary_text_encoder)
f.close()

# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                betas=(0.9, 0.999),
                                weight_decay=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.99)

# Initialize training
epoch = 0
loss = 0
contrastive = 0
triplet_loss_ = 0
contrastive_losses = []
losses = []
triplet_losses = []
cosineScores = []
sigmoidScores = []
score1 = 0
score2 = 0
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000


# Load a checkpoint
#save_path = os.path.join(SAVE_DIR, 'model_36.pt')
#checkpoint = torch.load(save_path)
#model.load_state_dict(checkpoint['model_state_dict'])


for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
    # Create DataLoader for the current fold
    train_fold_dataset = torch.utils.data.Subset(train_dataset, train_index)
    val_fold_dataset = torch.utils.data.Subset(train_dataset, val_index)
    train_fold_loader = TorchGeoDataLoader(train_fold_dataset, batch_size=batch_size, shuffle=True)
    val_fold_loader = TorchGeoDataLoader(val_fold_dataset, batch_size=batch_size, shuffle=True)

    # Initialize tensorboard for each fold
    writer = SummaryWriter(comment=COMMENT+'_fold'+str(fold+1))



    # Training loop

    for i in tqdm(range(nb_epochs)):
        print('-----EPOCH{} FOLD{}-----'.format(i+1, fold+1))
        model.train()
        for batch in train_fold_loader:
            #print(batch.batch)
            #print(batch.ptr)
            size = batch.num_graphs
            reference = torch.diag(torch.ones(size)).to(device)
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            if size > 1:
                cosineScore, sigmoidScore = scores(x_graph, x_text, reference, device=device, batch_size=size)
            else:
                cosineScore, sigmoidScore = 0, 0
        #contrastive_loss_ = contrastive_loss(x_graph, x_text)
            current_loss = contrastive_loss(x_graph, x_text)
            triplet_loss = BatchTripletLoss(x_text,x_graph, batch_size=size, device=device)
            
            optimizer.zero_grad()
            current_loss.backward()
            #triplet_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            triplet_loss_ += triplet_loss.item()
            loss += current_loss.item()
            #contrastive += contrastive_loss_.item()
            score1 += cosineScore.item()
            score2 += sigmoidScore.item()
            
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/printEvery))
                losses.append(loss)
                #contrastive_losses.append(contrastive)
                cosineScores.append(score1)
                sigmoidScores.append(score2)
                triplet_losses.append(triplet_loss_)
                #writer.add_scalar("NSCLoss/train", loss/printEvery, count_iter)
                writer.add_scalar("Loss/train", loss/printEvery, count_iter)
                writer.add_scalar("CosineScore/train", score1/printEvery, count_iter)
                writer.add_scalar("SigmoidScore/train", score2/printEvery, count_iter)
                writer.add_scalar("TripletLoss/train", triplet_loss_/printEvery, count_iter)
                #contrastive = 0
                loss = 0 
                score1 = 0
                score2 = 0
                triplet_loss_ = 0
                
                
        #Validation loop
        
        model.eval()       
        val_loss = 0
        contrastive_val_loss_ = 0        
        cosScore = 0
        sigScore = 0
        triplet_val_loss = 0
        for batch in val_fold_loader:
            size = batch.num_graphs
            reference = torch.diag(torch.ones(size)).to(device)
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            #contrastive_val_loss__ = contrastive_loss(x_graph, x_text)
            current_loss = contrastive_loss(x_graph, x_text)
            triplet_val_loss_ = BatchTripletLoss(x_text,x_graph, batch_size=size, device=device)
            if size > 1:
                cosineScore, sigmoidScore = scores(x_graph, x_text, reference, device=device, batch_size=size)
            else:
                cosineScore, sigmoidScore = 0, 0
            cosScore += cosineScore.item()
            sigScore += sigmoidScore.item()
            val_loss += current_loss.item()
            #contrastive_val_loss_ += contrastive_val_loss__.item()
            triplet_val_loss += triplet_val_loss_.item()
            
        
        best_validation_loss = min(best_validation_loss, val_loss)
        print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: '+  str(val_loss/len(val_fold_loader))+ ' - CosineScore: '+str(cosScore/len(val_fold_loader))+ ' - SigmoidScore: '+str(sigScore/len(val_fold_loader)))
        #writer.add_scalar("NSCLoss/validation", val_loss/len(val_loader), i)
        writer.add_scalar("Loss/validation", val_loss/len(val_fold_loader), i)
        writer.add_scalar("CosineScore/validation", cosScore/len(val_fold_loader), i)
        writer.add_scalar("SigmoidScore/validation", sigScore/len(val_fold_loader), i)
        writer.add_scalar("TripletLoss/validation", triplet_val_loss/len(val_fold_loader), i)
        
        if best_validation_loss==val_loss:
            print('validation loss improved, saving checkpoint...')
            save_path = os.path.join(SAVE_DIR, 'model_'+str(i)+'.pt')
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))

    writer.flush()
    writer.close()


# Write a summary of the training stats to a file
f = open(os.path.join(SAVE_DIR, 'training_summary.txt'), 'w')
f.write('Training summary\n')
f.write('================\n')
f.write('Presentation:\n')
f.write('Model: {}\n'.format(MODEL_SURNAME))
f.write('Timestamp: {}\n'.format(datetime.now()))
f.write('Host: {}\n'.format(gethostname()))
f.write('================\n')
f.write('Parameters:\n')
f.write('Number of epochs: {}\n'.format(nb_epochs))
f.write('Batch size: {}\n'.format(batch_size))
f.write('Learning rate: {}\n'.format(learning_rate))
f.write('================\n')
f.write('Model Parameters:\n')
f.write('Model name: {}\n'.format(model_name))
f.write('Number of parameters: {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
f.write('================\n')
f.write('Hardware:\n')
if device != torch.device("cpu"):
    f.write('GPU found: {}\n'.format(torch.cuda.get_device_name(0)))
    f.write('GPU memory: {:.3f} MB \n'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024))
else:
    f.write('No GPU found\n')
f.write('================\n')
time2 = time.time()
f.write('Training time: {}\n'.format(time2-time1))
f.write('Best validation loss: {}\n'.format(best_validation_loss/len(val_loader)))
f.write('Training loss mean over last 500 iterations: {}\n'.format(np.mean(losses[-10:])/printEvery))
f.write('================\n')
f.close()



# Submission

print('loading best model for submission...')
#save_path = os.path.join(SAVE_DIR, 'model_86.pt')
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='../data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='../data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = TorchGeoDataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

graph_embeddings = []
for batch in test_loader:
    for output in graph_model(batch.to(device)):
        graph_embeddings.append(output.tolist())

test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
text_embeddings = []
for batch in test_text_loader:
    for output in text_model(batch['input_ids'].to(device), 
                             attention_mask=batch['attention_mask'].to(device)):
        text_embeddings.append(output.tolist())



#similarity = cosine_similarity(text_embeddings, graph_embeddings)

similarity =sigmoid_kernel(text_embeddings, graph_embeddings)
#similarity = additive_chi2_kernel(text_embeddings, graph_embeddings)
solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(os.path.join(SUBMISSION_DIR, 'Sigmoidsubmission.csv'), index=False)
print('submission saved to: {}'.format(os.path.join(SUBMISSION_DIR, 'submission.csv')))
print('================ DONE ================')
similarity = cosine_similarity(text_embeddings, graph_embeddings)
solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(os.path.join(SUBMISSION_DIR, 'COSINEsubmission.csv'), index=False)
print('submission saved to: {}'.format(os.path.join(SUBMISSION_DIR, 'submission.csv')))
print('================ DONE ================')