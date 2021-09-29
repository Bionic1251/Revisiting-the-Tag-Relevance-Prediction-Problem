#!/usr/bin/env python
"""
IMPORTANT: INSTALL CPU VERSION PYTORCH
https://pytorch.org/get-started/locally/
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
"""
from loguru import logger
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import gensim
import gensim.downloader
from parameters import *
from sklearn.metrics import mean_absolute_error
logger.add("log_model_train.log")
from src.data.paths import *
from src.data.make_processed_train_test import *

fpath_data_survey = DIR_TRAIN_TEST # "/home/ms314/projects/movielens-tagnav/r/out/dataSurvey.csv"

TAG_EMBEDDINGS_DIM = 3

CROSS_ENTROPY_LOSS = "CrossEntropyLoss"
BCE_WITH_LOGITS_LOSS = "bce_with_logits_loss"
MSE_LOSS = "MSELoss"
L1_LOSS = "L1_Loss"

LOSS = MSE_LOSS
LOSS = CROSS_ENTROPY_LOSS
LOSS = BCE_WITH_LOGITS_LOSS
LOSS = L1_LOSS

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
#tags_word2vec =
tags_one_hot = None
# ENCODINGS = {WORD2VEC_ENCODING: tags_word2vec,
#              ONE_HOT_ENCODING: tags_one_hot}
#
# ENCODING = ONE_HOT_ENCODING
# ENCODING = WORD2VEC_ENCODING

# DIMENSION_ONE_HOT = 1071
# NUMBER_OF_INPUT_FEATURES = 8 + DIMENSION_ONE_HOT
DIMENSION_WORD2VEC = 300
NUMBER_OF_INPUT_FEATURES = 8 + DIMENSION_WORD2VEC


def create_dataloader(df):
    return torch.utils.data.DataLoader(TagnavDataset(df), batch_size=BATCH_SIZE, shuffle=False)


class TagnavDataset(torch.utils.data.Dataset):
    """ Input for constructor is Pandas data frame """
    def __init__(self, data):
        self.tag = data['tag']
        self.log_IMDB = data['log_IMDB']
        self.log_IMDB_nostem = data['log_IMDB_nostem']
        self.rating_similarity = data['rating_similarity']
        self.avg_rating = data['avg_rating']
        self.tag_exists = data['tag_exists']
        self.lsi_tags_75 = data['lsi_tags_75']
        self.lsi_imdb_175 = data['lsi_imdb_175']
        self.tag_prob = data['tag_prob']
        self.tag_encode = TagsMappingsW2V().load_tags_word2vec_mappings()
        self.targets = data['targets']

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
        try:
            tag_text_encoding = self.tag_encode[self.tag[idx]]
        except KeyError:
            logger.info(f"Embedding for tag: {self.tag[idx]} : is not pre-calculated")
            logger.info("Add it to embeddings dictionary")
            self.tag_encode[self.tag[idx]] = TagsMappingsW2V().calc_tag_embed(self.tag[idx])
            tag_text_encoding = self.tag_encode[self.tag[idx]]
        x = torch.cat((torch.tensor(tag_text_encoding), x))
        y = torch.tensor(self.targets[idx], dtype=torch.float) - 1.0
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


def train(train_data_loader,
          validation_loader=None,
          num_epochs=15,
          epoch_model_fname_prefix="_bm",
          load_model=None,
          load_checkpoint=None,
          save_model='model.bin',
          save_checkpoint=None,
          perf_json_save_path=os.path.join(PROJECT_DIR, 'temp/perf.json'),
          save_model_each_epoch=False,
          ):
    device = torch.device("cpu")
    if load_model:
        logger.info(f".. continue training from {load_model}")
        model = torch.load(load_model)
    else:
        model = Model().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    if load_checkpoint:
        logger.info(f".. load checkpoint {load_checkpoint}")
        checkpoint = torch.load(load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
    model.train()
    validation_losses = []
    performance = {"epoch": [], "train_loss": [], "val_loss": []}
    loss_value = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = 0
        batch_counter = 0
        for _, data in enumerate(train_data_loader):
            batch_counter += 1
            inputs, label = data
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            #
            # Training steps end
            total_batches += 1
            running_loss += loss.item()
            loss_value = loss.item()
        running_loss /= total_batches
        #if (epoch + 1) % 1 == 0:
        logger.info(f"Epoch = {epoch+1}, averaged {str(LOSS)} = {running_loss}, batches = {total_batches}")
        # if save_model_each_epoch:
        #     save_path_epoch_model = os.path.join(PROJECT_DIR, "out_bin/model_ep_" + str(epoch) + epoch_model_fname_prefix + ".bin")
        #     torch.save(model, save_path_epoch_model)
        # if validation_loader:
        #     logger.info(f"Run validation step")
        #     model.eval()
        #     running_loss_val = 0.0
        #     total_batches = 0
        #     for _, data in enumerate(validation_loader):
        #         inputs, label = data
        #         output = model(inputs)
        #         loss = criterion(output, label)
        #         total_batches += 1
        #         running_loss_val += loss.item()
        #     running_loss_val /= total_batches
        #     if len(validation_losses) > 1 and validation_losses[-1] < running_loss_val:
        #         diff = running_loss_val - validation_losses[-1]
        #         if 1 == 0:
        #             logger.info(f"... Validation loss has increased epoch={epoch}")
        #             logger.info(f"... Diff = {diff}")
        #             logger.info(f"... Diff % = {diff * 100 / validation_losses[-1]}")
        #             logger.info(f"!!! Use model from epoch {epoch-1}")
        #     validation_losses.append(running_loss_val)
        #     performance["epoch"].append(epoch)
        #     performance["train_loss"].append(running_loss)
        #     performance["val_loss"].append(running_loss_val)
        #     with open(perf_json_save_path, 'w') as f:
        #         json.dump(performance, f, indent=4)
        #     model.train()
    if save_model:
        torch.save(model, save_model)
    if save_checkpoint:
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_value,
                    },
                    save_checkpoint)


def eval_step(eval_dataset_loader,
              model_path_load='./out_bin/model.bin',
              save_predictions_to='./temp/predictions_temp.txt'):
    model_loaded = torch.load(model_path_load)
    logger.info(f"Load model from {model_path_load}")
    model_loaded.eval()
    predictions = []
    true_values = []
    for _, eval_data in enumerate(eval_dataset_loader):
        inputs_test, label_test = eval_data
        outputs_test = model_loaded(inputs_test)
        predictions.append(outputs_test)
        true_values.append(label_test / 4.0)

    result = None
    expected = None

    if LOSS == L1_LOSS or LOSS == MSE_LOSS:
        result = torch.cat(predictions, dim=0)
        expected = torch.cat(true_values, dim=0)
    elif LOSS == CROSS_ENTROPY_LOSS or LOSS == BCE_WITH_LOGITS_LOSS:
        # Put every tensor vector into 2D tensor row
        result = torch.cat([e.squeeze(0) for e in predictions], 0)
        expected = torch.cat([e.squeeze(0) for e in true_values], 0)

    loss = None
    if LOSS == L1_LOSS or LOSS == MSE_LOSS:
        loss = criterion(result, expected.double())
    elif LOSS == CROSS_ENTROPY_LOSS or LOSS == BCE_WITH_LOGITS_LOSS:
        loss = criterion(result, expected)

    result_numpy = None
    # logger.info(f"Test set {str(criterion)} = {loss}")
    result_numpy = result.detach().numpy()
    result_numpy = result_numpy * 4.0 + 1.0
    if save_predictions_to:
        np.savetxt(save_predictions_to, result_numpy, fmt='%1.2f',)
        logger.info(f"Save predictions into {save_predictions_to}")
    return result_numpy


def calc_loss_from_arrays(true_values, predictions, r_predictions_file=None):
    """ R: 0.78 """
    predictions_pytorch = torch.tensor(predictions, dtype=torch.float)
    true_values = torch.tensor(true_values, dtype=torch.float).unsqueeze(1)
    fn_loss = nn.L1Loss()
    mae = fn_loss(true_values, predictions_pytorch)
    if r_predictions_file:
        predictions_r_ = pd.read_csv(r_predictions_file, header=None)
        predictions_r_ = torch.tensor(predictions_r_.values, dtype=torch.float)
        logger.info(f"MAE from R predictions = {f(true_values, predictions_r_)}")
    return mae


def fn_mae(y_pred, y_true):
    test = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    assert mae == test
    return mae


def recalc_word2vec():
    trtst = TrainTestBooksMovies()
    df_train_m = trtst.train_movies
    df_train_b = trtst.train_books
    df_test = trtst.test_books
    all_tags = list(set(df_test['tag'].unique().tolist() +
                        df_train_b['tag'].unique().tolist() +
                        df_train_m['tag'].unique().tolist()))
    tags_map_obj = TagsMappingsW2V(all_tags)
    tags_map_obj.create_and_save_word2vec_tags_mapping()


class TagsMappingsW2V:
    output_file = os.path.join(PROJECT_DIR, 'temp/tags_word2vec.json')


    def create_and_save_word2vec_tags_mapping(self, tags_list):
        vectors = gensim.downloader.load('glove-wiki-gigaword-300')
        mapping = {}
        for tag in tags_list:
            mapping[tag] = self.calc_tag_embed(tag, vectors)
        with open(self.output_file, 'w') as f:
            json.dump(mapping, f)
        logger.info(f"Save into {self.output_file}")

    def calc_tag_embed(self, tag, gensim_vectors):
        for t in tag.split():
            embeddings_for_tag = []
            try:
                embeddings_for_tag.append(gensim_vectors[t])
            except KeyError:
                embeddings_for_tag.append(np.ones((300,)) * 0.5)
        return np.mean(embeddings_for_tag, axis=0).tolist()

    def load_tags_word2vec_mappings(self):
        with open(self.output_file, 'r') as f:
            return json.load(f)


def run_on_single_split():
    trtst = TrainTestBooksMovies(books_frac=None)
    if 1 == 0:
        trtst.make_train_test(make_ten_folds=False)
    else:
        logger.info(f"Do not redo train test sets")
    df_train_m = trtst.train_movies
    df_train_b = trtst.train_books
    df_train_bm = trtst.train_books_and_movies
    df_test = trtst.test_books

    dataloader_train_b = create_dataloader(df_train_b)
    dataloader_train_m = create_dataloader(df_train_m)
    dataloader_train_bm = create_dataloader(df_train_bm)

    logger.info(f"Shape df_rain_m: {df_train_m.shape}")
    logger.info(f"Shape df_train_b: {df_train_b.shape}")
    logger.info(f"Shape df_test_b: {df_test.shape}")

    file_pred_b = os.path.join(PROJECT_DIR, 'temp/predictions_only_books.txt')
    file_pred_bm = os.path.join(PROJECT_DIR, 'temp/predictions_books_and_movies.txt')
    file_pred_bm_same_time = os.path.join(PROJECT_DIR, 'temp/predictions_books_and_movies_same_time.txt')
    file_true_values = os.path.join(PROJECT_DIR, 'temp/true_values.txt')

    df_test['targets'].to_csv(file_true_values, header=None, index=False)

    model_b_bin = os.path.join(PROJECT_DIR, 'out_bin/model_b.bin')
    model_m_bin = os.path.join(PROJECT_DIR, 'out_bin/model_m.bin')
    model_bm_bin = os.path.join(PROJECT_DIR, 'out_bin/model_bm.bin')
    model_bm_same_time_bin = os.path.join(PROJECT_DIR, 'out_bin/model_bm_same_time.bin')

    checkpoint_file = os.path.join(PROJECT_DIR, 'out_bin/checkpoint.pt')


    def pretrain_model_on_movies(n_epochs_movies=30):
        logger.info(f"Pre-train model on Movies")
        train(dataloader_train_m,
              num_epochs=n_epochs_movies,
              save_model=model_m_bin,
              save_checkpoint=checkpoint_file)
        logger.info(f"Save pre-trained model into {model_b_bin}")
        logger.info(f"Save pre-trained model checkpoint into {checkpoint_file}")


    def train_on_books(n_epochs_books=10):
        logger.info(".. train on books only")
        train(dataloader_train_b,
              num_epochs=n_epochs_books,
              save_model=model_b_bin)
        logger.info(f"Save model trained on books into {model_b_bin}")
        logger.info(".. train on books and movies")
        train(dataloader_train_b,
              num_epochs=n_epochs_books,
              load_model=model_m_bin,
              load_checkpoint=checkpoint_file,
              save_model=model_bm_bin)
        logger.info(f"Save model trained on books and movies into {model_bm_bin}")
        #logger.info(".. train on books and movies at the same time")
        #train(dataloader_train_bm,
        #      num_epochs=n_epochs_books,
        #      save_model=model_bm_same_time_bin)
        #logger.info(f"Save model trained on books and movies at the same time into {model_bm_same_time_bin}")

    def run_test():
        # books
        eval_step(create_dataloader(df_test),
                  model_path_load=model_b_bin,
                  save_predictions_to=file_pred_b)
        # books and movies sequential
        eval_step(create_dataloader(df_test),
                  model_path_load=model_bm_bin,
                  save_predictions_to=file_pred_bm)
        ## books and movies same time
        #eval_step(create_dataloader(df_test),
        #          model_path_load=model_bm_same_time_bin,
        #          save_predictions_to=file_pred_bm_same_time)

    def calc_mae_from_files():
        predictions_b = pd.read_csv(file_pred_b, header=None).to_numpy()
        predictions_bm = pd.read_csv(file_pred_bm, header=None).to_numpy()
        predictions_bm_same_time = pd.read_csv(file_pred_bm_same_time, header=None).to_numpy()
        true_values = df_test.targets.to_numpy()
        mae_b = calc_loss_from_arrays(true_values, predictions_b)
        mae_bm = calc_loss_from_arrays(true_values, predictions_bm)
        mae_bm_same_time = calc_loss_from_arrays(true_values, predictions_bm_same_time)
        logger.info(f"MAE Books            = {mae_b}")
        logger.info(f"MAE Books and Movies = {mae_bm}")
        logger.info(f"MAE Books and Movies same time = {mae_bm_same_time}")

    if 1 == 0:
        pretrain_model_on_movies(n_epochs_movies=30)
    else:
        logger.info(f"Do not re-calculate pre-trained model")
    train_on_books(n_epochs_books=10)
    run_test()
    calc_mae_from_files()


def run_on_ten_folds():
    trtst = TrainTestBooksMovies()
    df_train_m = trtst.train_movies
    if 1 == 0:
       trtst.make_train_test(make_ten_folds=True)

    model_b_bin = os.path.join(PROJECT_DIR, 'temp/model_b.bin')
    model_m_bin = os.path.join(PROJECT_DIR, 'temp/model_m.bin')
    model_bm_bin = os.path.join(PROJECT_DIR, 'temp/model_bm.bin')
    checkpoint_file = os.path.join(PROJECT_DIR, 'temp/checkpoint.pt')

    n_epochs_only_books = 150
    n_epochs_movies = 50
    n_epochs_books_after_movies = 100

    if 1 == 1:
        logger.info("Pre-train model on Movies ")
        train(create_dataloader(df_train_m),
              num_epochs=n_epochs_movies,
              save_model=model_m_bin,
              save_checkpoint=checkpoint_file)
    else:
        logger.info("Model for Movies is not re-calculated")

    for fold in range(1, 11):
        logger.info(f"Fold {fold}")
        train_data_loader = None
        test_data_loader = None
        try:
            df_train_b, df_test_b = trtst.get_train_test_books_fold(fold)
            logger.info(f".. {df_train_b.shape}")
            logger.info(f".. {df_test_b.shape}")
            logger.info(".. create dataloaders")
            train_data_loader = create_dataloader(df_train_b)
            test_data_loader = create_dataloader(df_test_b)
            logger.info(".. finished")
        except FileNotFoundError:
            logger.info(f"Was trying to read KFold data sets for fold {fold}")

        logger.info(".. train only on books")
        train(train_data_loader,
              num_epochs=n_epochs_only_books,
              save_model=model_b_bin)

        logger.info(".. train only on books and movies")
        train(train_data_loader,
              num_epochs=n_epochs_books_after_movies,
              load_model=model_m_bin,
              load_checkpoint=checkpoint_file,
              save_model=model_bm_bin)
        logger.info(".. finished")

        prediction_b = eval_step(test_data_loader,
                                 model_path_load=model_b_bin,
                                 save_predictions_to=None)

        predictions_bm = eval_step(test_data_loader,
                                   model_path_load=model_bm_bin,
                                   save_predictions_to=None)

        true_values = df_test_b['targets'].to_numpy()
        mae_b = calc_loss_from_arrays(true_values, prediction_b)
        mae_bm = calc_loss_from_arrays(true_values, predictions_bm)
        logger.info(f"Fold {fold}")
        logger.info(f"MAE Books            = {mae_b}")
        logger.info(f"MAE Books and Movies = {mae_bm}")


if __name__ == "__main__":
    run_on_ten_folds()
