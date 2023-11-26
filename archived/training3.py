###
#IMPORT DEPENDENCIES
###
from init import *

use_gpu = True
use_dataparallel = True
device = 'cuda'

sys.path.insert(0, '../..')

torch.manual_seed(42)

IMAGE_WIDTH = {5: 15, 20: 60, 60: 180}
IMAGE_HEIGHT = {5: 32, 20: 64, 60: 96}

####
#LOAD DATA
####

year_list = np.arange(1993, 2001, 1)

images = []
label_df = []
for year in year_list:
    images.append(
        np.memmap(os.path.join("../img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_images.dat"), dtype=np.uint8,
                  mode='r').reshape(
            (-1, IMAGE_HEIGHT[20], IMAGE_WIDTH[20])))
    label_df.append(pd.read_feather(
        os.path.join("../img_data/monthly_20d", f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather")))
    print(f'year {year} done!')

images = np.concatenate(images)
label_df = pd.concat(label_df)

###
#Build Dataset
###

class MyDataset(Dataset):

    def __init__(self, img, label):
        self.img = torch.Tensor(img.copy())
        self.label = torch.Tensor(label)
        self.len = len(img)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

###
#Split Method with I20/5R
###


# Use 70%/30% ratio for train/validation split
train_indices, val_indices = train_test_split(
    np.arange(images.shape[0]),
    test_size=0.3,
    random_state=42
)

# Create datasets based on the selected indices
train_dataset = MyDataset(images[train_indices], (label_df.Ret_5d > 0).values[train_indices])
val_dataset = MyDataset(images[val_indices], (label_df.Ret_5d > 0).values[val_indices])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)


###
#Model Weights
###
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)


print(f'YOU ARE USING THE {device}')

export_onnx = True
net = Net().to(device)
net.apply(init_weights)

if export_onnx:
    import torch.onnx
    x = torch.randn([1, 1, 64, 60]).to(device)
    torch.onnx.export(net,               # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      "./cnn_baseline.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=False,        # store the trained parameter weights inside the model file
                      opset_version=10,          # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names=['input_images'],   # the model's input names
                      output_names=['output_prob'], # the model's output names
                      dynamic_axes={'input_images': {0: 'batch_size'},    # variable length axes
                                     'output_prob': {0: 'batch_size'}})


    ###
    #Profiling
    ###

    count = 0
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())
        count += parameters.numel()
    print('total_parameters : {}'.format(count))



    flops, params = thop_profile(net, inputs=(next(iter(train_dataloader))[0].to(device),))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

    from torch.profiler import profile, record_function, ProfilerActivity

    inputs = next(iter(train_dataloader))[0].to(device)

    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            net(inputs)

    prof.export_chrome_trace("../trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    ###
    #Training
    ###


def train_loop(dataloader, net, loss_fn, optimizer):
    running_loss = 0.0
    current = 0
    net.train()

    with tqdm(dataloader) as t:
        for batch, (X, y) in enumerate(t):
            X = X.to(device)
            y = y.to(device)
            y_pred = net(X)

            if y == 1:
                y = torch.tensor([0, 1]).unsqueeze(0)
                y = y.to(torch.float32)
            else:
                y = torch.tensor([1, 0]).unsqueeze(0)
                y = y.to(torch.float32)

            y = y.to(device)

            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = (len(X) * loss.item() + running_loss * current) / (len(X) + current)
            current += len(X)
            t.set_postfix({'running_loss': running_loss})

    return running_loss


def val_loop(dataloader, net, loss_fn):
    running_loss = 0.0
    current = 0
    net.eval()

    with torch.no_grad():
        with tqdm(dataloader) as t:
            for batch, (X, y) in enumerate(t):
                X = X.to(device)
                y = y.to(device)
                y_pred = net(X)

                if y == 1:
                    y = torch.tensor([0, 1]).unsqueeze(0)
                    y = y.to(torch.float32)
                else:
                    y = torch.tensor([1, 0]).unsqueeze(0)
                    y = y.to(torch.float32)

                y = y.to(device)

                loss = loss_fn(y_pred, y)

                running_loss += loss.item()
                running_loss = (len(X) * running_loss + loss.item() * current) / (len(X) + current)
                current += len(X)

    return running_loss


if use_gpu and use_dataparallel and 'DataParallel' not in str(type(net)):
    net = net.to(device)
    net = nn.DataParallel(net)


loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
start_epoch = 0
min_val_loss = 1e9
last_min_ind = -1
early_stopping_epoch = 5
epochs = 100
tb = SummaryWriter()


###
#EXECUTE TRAIN AND WRITE LOGS
###

start_time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')

logs_folder = os.path.join('../CNN/losses', f'logs_{start_time}')
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder)


for t in range(start_epoch, epochs):
    print(f"Epoch {t}\n-------------------------------")
    time.sleep(0.2)
    train_loss = train_loop(train_dataloader, net, loss_fn, optimizer)
    val_loss = val_loop(val_dataloader, net, loss_fn)
    tb.add_histogram("train_loss", train_loss, t)

    # DIR check and save epoch
    save_dir = os.path.join('../CNN/models', f'baseline_epoch_{t}_train_{train_loss:.5f}_val_{val_loss:.5f}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.pt')
    torch.save(net, save_path)

    # Early stopping
    if val_loss < min_val_loss:
        last_min_ind = t
        min_val_loss = val_loss
    elif t - last_min_ind >= early_stopping_epoch:
        break

print('Done!')
print('Best epoch: {}, val_loss: {}'.format(last_min_ind, min_val_loss))