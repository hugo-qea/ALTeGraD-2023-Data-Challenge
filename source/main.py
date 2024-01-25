from utils import *
from models.Model import Baseline, ModelAttention, ModelSAGE, ModelGATConv, ModelAttentiveFP, ModelGATPerso, ModelGATwMLP, AttentionEncoder
from torch.utils.tensorboard import SummaryWriter



# Reproducibility
seed = 13
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
#torch.use_deterministic_algorithms(True)


# Setup the text encoder
#model_name = 'distilbert-base-uncased'
#model_name= 'distilbert-base-uncased-finetuned-sst-2-english'
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
model_name = 'dmis-lab/biobert-v1.1'
model_name = 'microsoft/deberta-v3-base'
model_name = 'DeepChem/ChemBERTa-10M-MLM'
model_name = 'DeepChem/SmilesTokenizer_PubChem_1M'
model_name = 'unikei/bert-base-smiles'

tokenizer = AutoTokenizer.from_pretrained(model_name)


gt = np.load("../data/token_embedding_dict.npy", allow_pickle=True)[()]
val_dataset = GraphTextDataset(root='../data/', gt=gt, split='val', tokenizer=tokenizer)
train_dataset = GraphTextDataset(root='../data/', gt=gt, split='train', tokenizer=tokenizer)

# Train on GPU if possible (it is actually almost mandatory considering the size of the model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if device != torch.device("cpu"):
    print('================ GPU FOUND ================')
    print('GPU found: {}'.format(torch.cuda.get_device_name(0)))
    print('GPU memory: {:.3f} MB'.format(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024))
else:
    print('================ NO GPU ================')

# Training hyperparameters
nb_epochs = 5
batch_size = 16
learning_rate = 5e-5

# Setup the batch loaders
val_loader = TorchGeoDataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = TorchGeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#model = Baseline(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
model = ModelAttention(model_name=model_name, n_in=300, nout=768, nhid=1024, attention_hidden=2048, dropout=0.3) # nout = bert model hidden dim
#model = ModelSAGE(model_name=model_name, n_in=300, nout=768, nhid=1000, sage_hidden=1000, dropout=0.3) # nout = bert model hidden dim
#model = ModelGATConv(model_name=model_name, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.3) # nout = bert model hidden dim
#model = ModelAttentiveFP(model_name=model_name, n_in=300, nout=768, nhid=1000, attention_hidden=1000, dropout=0.3) # nout = bert model hidden dim
#model = ModelGATPerso(model_name=model_name, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.6) # nout = bert model hidden dim
#model = ModelGATwMLP(model_name=model_name, n_in=300, nout=768, nhid=1024, n_heads=8, dropout=0.6) # nout = bert model hidden dim
model.to(device)

MODEL_SURNAME =  model.get_model_surname() + '_TEST_' + model_name
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
                                weight_decay=0.01)

# Initialize training
epoch = 0
loss = 0
losses = []
count_iter = 0
time1 = time.time()
printEvery = 50
best_validation_loss = 1000000

# Initialize tensorboard
writer = SummaryWriter(comment=COMMENT)

# Training loop


for i in tqdm(range(nb_epochs)):
    print('-----EPOCH{}-----'.format(i+1))
    model.train()
    for batch in train_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        loss += current_loss.item()
        
        count_iter += 1
        if count_iter % printEvery == 0:
            time2 = time.time()
            print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                        time2 - time1, loss/printEvery))
            losses.append(loss)
            writer.add_scalar("Loss/train", loss/printEvery, count_iter)
            loss = 0 
    model.eval()       
    val_loss = 0        
    for batch in val_loader:
        input_ids = batch.input_ids
        batch.pop('input_ids')
        attention_mask = batch.attention_mask
        batch.pop('attention_mask')
        graph_batch = batch
        x_graph, x_text = model(graph_batch.to(device), 
                                input_ids.to(device), 
                                attention_mask.to(device))
        current_loss = contrastive_loss(x_graph, x_text)   
        val_loss += current_loss.item()
    best_validation_loss = min(best_validation_loss, val_loss)
    print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
    writer.add_scalar("Loss/validation", val_loss/len(val_loader), i)
    if best_validation_loss==val_loss:
        print('validation loss improved saving checkpoint...')
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
checkpoint = torch.load(save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

graph_model = model.get_graph_encoder()
text_model = model.get_text_encoder()

test_cids_dataset = GraphDataset(root='../data/', gt=gt, split='test_cids')
test_text_dataset = TextDataset(file_path='../data/test_text.txt', tokenizer=tokenizer)

idx_to_cid = test_cids_dataset.get_idx_to_cid()

test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

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



similarity = cosine_similarity(text_embeddings, graph_embeddings)

solution = pd.DataFrame(similarity)
solution['ID'] = solution.index
solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
solution.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)
print('submission saved to: {}'.format(os.path.join(SUBMISSION_DIR, 'submission.csv')))
print('================ DONE ================')