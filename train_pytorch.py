"""
IMPORTANT: INSTALL CPU VERSION PYTORCH
https://pytorch.org/get-started/locally/
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from utils_plots import *
from train_pytorch_settings import *
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()
from sklearn.metrics import mean_absolute_error

from data import *

fpath_data_survey = "movielens-tagnav/r/out/dataSurvey.csv"
fpath_y = "movielens-tagnav/r/out/y.csv"
R_PREDICTIONS_FILE = "movielens-tagnav/r/out/surveyPredictions.csv"

TAG_EMBEDDINGS_DIM = 3

CROSS_ENTROPY_LOSS = "CrossEntropyLoss"
BCE_WITH_LOGITS_LOSS = "bce_with_logits_loss"
MSE_LOSS = "MSELoss"
L1_LOSS = "L1_Loss"

LOSS = MSE_LOSS
LOSS = CROSS_ENTROPY_LOSS
LOSS = BCE_WITH_LOGITS_LOSS
LOSS = L1_LOSS

logger.info(f"Use {str(LOSS)} loss function")

LOSS_FUNCTIONS = {L1_LOSS: nn.L1Loss(),
                  MSE_LOSS: nn.MSELoss(),
                  CROSS_ENTROPY_LOSS: nn.CrossEntropyLoss(),
                  BCE_WITH_LOGITS_LOSS: nn.BCEWithLogitsLoss(),
                  }

criterion = LOSS_FUNCTIONS[LOSS]


def transform_to_one_hot(val):
    if int(val) not in range(5):
        raise TypeError
    else:
        encoded = torch.zeros(5)
        encoded[int(val)] = 1
        return encoded


def transform_one_hot_to_index(vec):
    return torch.argmax(vec).item() + 1


def get_train_test(frac=0.8):
    train_s = data_survey.sample(frac=frac, random_state=8)
    test_s = data_survey.drop(train_s.index).reset_index(drop=True)
    train_s = train_s.reset_index(drop=True)
    return train_s, test_s


WORD2VEC_ENCODING = "word2vec"
ONE_HOT_ENCODING = "one_hot"
ENCODINGS = {WORD2VEC_ENCODING: tags_word2vec,
             ONE_HOT_ENCODING: tags_one_hot}

ENCODING = WORD2VEC_ENCODING
ENCODING = ONE_HOT_ENCODING

DIMENSION_ONE_HOT = 1071
DIMENSION_WORD2VEC = 300
NUMBER_OF_INPUT_FEATURES = 8 + DIMENSION_WORD2VEC
NUMBER_OF_INPUT_FEATURES = 8 + DIMENSION_ONE_HOT


class TagnavDataset(torch.utils.data.Dataset):
    """ Input for constructor is Pandas data frame """
    def __init__(self, data):
        self.tag = data.tag
        self.log_IMDB = data.log_IMDB
        self.log_IMDB_nostem = data.log_IMDB_nostem
        self.rating_similarity = data.rating_similarity
        self.avg_rating = data.avg_rating
        self.tag_exists = data.tag_exists
        self.lsi_tags_75 = data.lsi_tags_75
        self.lsi_imdb_175 = data.lsi_imdb_175
        self.tag_prob = data.tag_prob
        self.targets = data.targets

    @classmethod
    def from_csv_file(cls, file_path):
        return cls(pd.read_csv(file_path, header=True, sep=","))

    def __getitem__(self, idx):
        x = torch.tensor([self.log_IMDB[idx],
                          self.log_IMDB_nostem[idx],
                          self.rating_similarity[idx],
                          self.avg_rating[idx],
                          self.tag_exists[idx],
                          self.lsi_tags_75[idx],
                          self.lsi_imdb_175[idx],
                          self.tag_prob[idx],
                          ], dtype=torch.float, requires_grad=True)
        tag_text_encoding = ENCODINGS[ENCODING][self.tag[idx]]
        x = torch.cat((torch.tensor(tag_text_encoding), x))
        # Minus 1 for CrossEntropy loss and for scaling by 4.0 for L1 and L2
        y = torch.tensor(self.targets[idx], dtype=torch.float) - 1.0
        if LOSS == CROSS_ENTROPY_LOSS:
            return x, torch.unsqueeze(y, 0).squeeze().long()
        elif LOSS == BCE_WITH_LOGITS_LOSS:
            return x, transform_to_one_hot(y)
        elif LOSS == L1_LOSS or LOSS == MSE_LOSS:
            y = y / 4.0
            return x, torch.unsqueeze(y, 0)

    def __len__(self):
        return len(self.targets)


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(NUMBER_OF_INPUT_FEATURES, HIDDEN_LAYER_SIZE)
        self.l2 = nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
        self.dropout = torch.nn.Dropout(0.3)
        if LOSS == CROSS_ENTROPY_LOSS:
            self.predict = nn.Linear(HIDDEN_LAYER_SIZE, NUMBER_OF_INPUT_FEATURES)  # todo: why?
        elif LOSS == BCE_WITH_LOGITS_LOSS:
            self.predict = nn.Linear(HIDDEN_LAYER_SIZE, 5)
        elif LOSS == L1_LOSS or LOSS == MSE_LOSS:
            self.predict = nn.Linear(HIDDEN_LAYER_SIZE, 1)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)

        x = self.l2(x)
        x = F.relu(x)

        x = self.l2(x)
        x = F.relu(x)

        x = self.predict(x)
        return x


def create_dataloader(df):
    return torch.utils.data.DataLoader(TagnavDataset(df), batch_size=BATCH_SIZE, shuffle=False)


def create_train_validation_dataloader(df):
    indices = range(len(df))
    train_idx, val_idx = split_indices_train_validation(indices, validation_split=VALIDATION_SPLIT)
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(TagnavDataset(df), batch_size=BATCH_SIZE, shuffle=False,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(TagnavDataset(df), batch_size=BATCH_SIZE, shuffle=False,
                                                    sampler=validation_sampler)
    return train_loader, validation_loader


def train(train_data_loader,
          validation_loader=None,
          num_epochs=15,
          model_path_save='./out_bin/model.bin',
          perf_json_save_path='./temp/perf.json'):
    """ https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html """
    device = torch.device("cpu")
    model = Model().to(device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    model.train()
    validation_losses = []
    performance = {"epoch": [],
                   "train_loss": [],
                   "val_loss": []}
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = 0
        for i, data in enumerate(train_data_loader):
            inputs, label = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, label)
            # writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            #
            # Training steps end
            total_batches += 1
            running_loss += loss.item()
        running_loss /= total_batches
        # print(f"Epoch = {epoch+1}, averaged {str(LOSS)} = {running_loss}, batches = {total_batches}")
        save_path_epoch_model = os.path.join("out_bin", "model_ep_" + str(epoch) + ".bin")
        torch.save(model, save_path_epoch_model)
        # logger.info(f"Save model {save_path_epoch_model}")
        if validation_loader:
            model.eval()
            running_loss_val = 0.0
            total_batches = 0
            for i, data in enumerate(validation_loader):
                inputs, label = data
                output = model(inputs)
                loss = criterion(output, label)
                total_batches += 1
                running_loss_val += loss.item()
            running_loss_val /= total_batches
            if len(validation_losses) > 1 and validation_losses[-1] < running_loss_val:
                diff = running_loss_val - validation_losses[-1]
                if 1 == 1:
                    print(f"... Validation loss has increased epoch={epoch}")
                    print(f"... Diff = {diff}")
                    print(f"... Diff % = {diff * 100 / validation_losses[-1]}")
                    print(f"!!! Use model from epoch {epoch-1}")
            validation_losses.append(running_loss_val)
            performance["epoch"].append(epoch)
            performance["train_loss"].append(running_loss)
            performance["val_loss"].append(running_loss_val)
            save_json(performance, perf_json_save_path)
            plot_performance_metrics(perf_json_save_path, './img/perf_fold_' + str(i) + '.PNG')
            model.train()
    torch.save(model, model_path_save)
    logger.info(f"Save model into {model_path_save}")
    logger.info('Finished Training')


def eval_step(eval_dataset_loader,
              model_path_load='./out_bin/model.bin',
              save_predictions_to='./temp/predictions_temp.txt'):
    model_loaded = torch.load(model_path_load)
    logger.info(f"Load model from {model_path_load}")
    model_loaded.eval()
    predictions = []
    true_values = []
    for i, eval_data in enumerate(eval_dataset_loader):
        inputs_test, label_test = eval_data
        outputs_test = model_loaded(inputs_test)
        predictions.append(outputs_test)
        if LOSS == MSE_LOSS or LOSS == L1_LOSS:
            true_values.append(label_test / 4.0)  # bcs in loader we do (val-1.0)
        else:
            true_values.append(label_test)

    test_criterion = criterion(predictions[0], true_values[0])
    logger.info(f"Test criterion calculation: {test_criterion}")

    result = None
    expected = None

    if LOSS == L1_LOSS or LOSS == MSE_LOSS:
        result = torch.cat(predictions, dim=0)
        expected = torch.cat(true_values, dim=0)
    elif LOSS == CROSS_ENTROPY_LOSS or LOSS == BCE_WITH_LOGITS_LOSS:
        # Put every tensor vector into 2D tensor row
        result = torch.cat([e.squeeze(0) for e in predictions], 0)
        expected = torch.cat([e.squeeze(0) for e in true_values], 0)

    logger.info(f"result.size={result.size()}")
    logger.info(f"expected.size={expected.size()}")

    loss = None
    if LOSS == L1_LOSS or LOSS == MSE_LOSS:
        loss = criterion(result, expected.double())
    elif LOSS == CROSS_ENTROPY_LOSS or LOSS == BCE_WITH_LOGITS_LOSS:
        loss = criterion(result, expected)

    result_numpy = None
    logger.info(f"Test set {str(criterion)} = {loss}")
    if LOSS == L1_LOSS or LOSS == MSE_LOSS:
        result_numpy = result.detach().numpy()
        result_numpy = result_numpy * 4.0 + 1.0
    elif LOSS == CROSS_ENTROPY_LOSS or LOSS == BCE_WITH_LOGITS_LOSS:
        result_numpy = torch.argmax(result, dim=1).detach()
        result_numpy = result_numpy + 1.0
    np.savetxt(save_predictions_to, result_numpy, fmt='%1.2f',)
    logger.info(f"Save predictions into {save_predictions_to}")


def calc_loss_from_saved_files(true_values_df_columns_values, predictions_file='', r_predictions_file=None):
    """ R: 0.78 """
    predictions_pytorch = pd.read_csv(predictions_file, header=None)
    predictions_pytorch = predictions_pytorch.values
    predictions_pytorch = torch.tensor(predictions_pytorch, dtype=torch.float)
    true_values = torch.tensor(true_values_df_columns_values, dtype=torch.float).unsqueeze(1)
    f = nn.L1Loss()
    print(f"MAE from PyTorch predictions = {f(true_values, predictions_pytorch)}")
    if r_predictions_file:
        predictions_r_ = pd.read_csv(r_predictions_file, header=None)
        predictions_r_ = torch.tensor(predictions_r_.values, dtype=torch.float)
        print(f"MAE from R predictions = {f(true_values, predictions_r_)}")


class TenFoldsData:
    ten_folds_dir = "libdat/data/10_folds"

    def train_validation_loader(self, i):
        train_loader, validation_loader = create_train_validation_dataloader(self.train_df(i))
        return train_loader, validation_loader

    def test_loader(self, i):
        return create_dataloader(self.test_df(i))

    def train_df(self, i):
        return self.read(i, 'train')

    def test_df(self, i):
        return self.read(i, 'test')

    def read(self, i, prefix='train'):
        train_path = os.path.join(self.ten_folds_dir, prefix + str(i) + ".csv")
        if os.path.exists(train_path):
            train_data = pd.read_csv(train_path)
            return train_data


tenfolds = TenFoldsData()


def run_train_eval():
    for i in range(10):
        train_loader, validation_loader = tenfolds.train_validation_loader(i)
        test_loader = tenfolds.test_loader(i)
        model_path = './out_bin/model_fold_' + str(i) + '.bin'
        train(train_loader,
              validation_loader=validation_loader,
              num_epochs=NUM_EPOCHS,
              model_path_save=model_path,
              perf_json_save_path='./temp/perf_fold_' + str(i) + '.json')

        eval_step(test_loader,
                  model_path_load=model_path,
                  save_predictions_to='./temp/predictions_fold_' + str(i) + '.txt')




def fn_mae(y_pred, y_true):
    test = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == test
    return mae


def calc_mae_for_ten_fold():
    temp_folder_path = 'temp'
    for i in range(10):
        test_df = tenfolds.test_df(i)
        train_df = tenfolds.train_df(i)
        mu_train = np.mean(train_df.targets.values)
        y_true = list(test_df.targets.values)
        mae_r = None
        try:
            r_predictions_path = "libdat/temp/r_predictions_fold_" + str(i) + ".txt"
            r_predictions = pd.read_csv(r_predictions_path, header=None)
            r_predictions = [e[0] for e in r_predictions.values]
            mae_r = fn_mae(r_predictions, y_true)
        except FileNotFoundError:
            pass
        predictions_path = "libdat/temp/predictions_fold_" + str(i) + ".txt"
        predictions = pd.read_csv(predictions_path, header=None)
        predictions = [e[0] for e in predictions.values]
        mae = fn_mae(predictions, y_true)
        mae_baseline = fn_mae(y_true, [mu_train]*len(y_true))
        # todo: save to file report as paper results
        if mae_r:
            print("MAE {0:1.3} Baseline MAE {1:1.3f} R MAE {2:1.3f}".format(mae, mae_baseline, mae_r))
        else:
            print("MAE {0:1.3} Baseline MAE {1:1.3f} R MAE - not ready".format(mae, mae_baseline))


if __name__ == "__main__":
    if 1 == 0:
        run_train_eval()
    calc_mae_for_ten_fold()
