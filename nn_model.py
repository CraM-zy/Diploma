import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
from collections import OrderedDict
from sqlalchemy import create_engine

tqdm.pandas()

torch.manual_seed(101)

def transform_df(items_dataset):

    items_dataset['datetime'] = pd.to_datetime(items_dataset['datetime']).dt.date

    items_dataset['ageRestricted'] = bool_map(items_dataset['ageRestricted'])
    items_dataset['containsAlcohol'] = bool_map(items_dataset['containsAlcohol'])
    items_dataset['minShelfLife'] = pd.to_numeric(items_dataset['minShelfLife'], errors='coerce').fillna(0)

    items_dataset = items_dataset.groupby(['datetime', 'warehouse', 'sku'], as_index=False) \
        .agg({'qty': 'sum', 'price': 'mean',
              'ageRestricted': 'mean', 'containsAlcohol': 'mean',
              'minShelfLife': 'mean',
              'VAT': 'mean'
              })

    items_dataset['warehouse_1001'] = (items_dataset['warehouse'] == '1001').astype('float')
    items_dataset['warehouse_1002'] = (items_dataset['warehouse'] == '1002').astype('float')

    prices = items_dataset[['datetime', 'warehouse', 'sku', 'price', 'ageRestricted', 'containsAlcohol',
                            'warehouse_1001', 'warehouse_1002', 'minShelfLife', 'VAT']]

    items_dataset = items_dataset.drop(columns=['price', 'ageRestricted', 'containsAlcohol', 'minShelfLife', 'VAT',
                                                'warehouse_1001', 'warehouse_1002']).drop_duplicates() \
        .pivot(index='datetime', columns=['warehouse', 'sku'], values='qty')
    items_dataset = items_dataset.asfreq('D').fillna(0)
    items_dataset = pd.melt(items_dataset.reset_index(), id_vars=['datetime'])
    items_dataset['sku'] = items_dataset['sku'].str.replace("-", "_")
    items_dataset['datetime'] = pd.to_datetime(items_dataset['datetime']).dt.date

    items_dataset = items_dataset.merge(prices, on=['datetime', 'warehouse', 'sku'], how='left')

    for variable in ["price", "VAT", 'minShelfLife', 'ageRestricted', 'containsAlcohol', 'warehouse_1001',
                     'warehouse_1002']:
        items_dataset[variable] = items_dataset.groupby(['warehouse', 'sku'])[variable].transform(
            lambda x: x.fillna(method="ffill").fillna(method="bfill").fillna(0))

    items_dataset['price'] = np.log1p(items_dataset['price'])
    items_dataset['VAT'] = items_dataset['VAT'] / 20
    items_dataset['minShelfLife'] = items_dataset['minShelfLife'] / 365.25
    items_dataset['weekday'] = pd.to_datetime(items_dataset['datetime']).dt.weekday / 7
    return items_dataset


class SequenceDataset(Dataset):
    def __init__(self, items_dataset, train_period=28, test_period=14, train=True):
        self.train = train
        self.train_period = train_period
        self.test_period = test_period
        if train:
            self.step = 25
        else:
            self.step = 1
        self.ds = self.seq_set(items_dataset)

    def to_array(self, dt):
        if self.train:
            self.step = random.randint(14, 42)
        ds = []

        for i in range(random.randint(0, dt.shape[0] - self.train_period - self.test_period + 1),
                       dt.shape[0] - self.train_period - self.test_period + 1, self.step):
            ds.append((
                torch.from_numpy(dt['value'][i:(i + self.train_period)].values.reshape(-1, 1)),
                torch.from_numpy(dt['value'][(i + self.train_period):(i + self.train_period + self.test_period)]
                                 .values.reshape(-1, 1)),
                torch.from_numpy(dt[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'humidity', 'wind_speed', 'pressure', 'weekday']]
                                 .iloc[(i + self.train_period):(i + self.train_period + self.test_period), :]
                                 .values),
                torch.from_numpy(dt[['price', 'ageRestricted', 'containsAlcohol', 'warehouse_1001',
                                     'warehouse_1002', 'minShelfLife', 'VAT']].iloc[0, :].values)
            ))
        return ds

    def seq_set(self, items_dataset):
        ds = []
        pairs = items_dataset.groupby(['warehouse', 'sku'])
        g = pairs.progress_apply(lambda x: self.to_array(x))
        for i in g.values:
            ds.extend(i)
        return ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


class ForecasterModel(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.input_dim = 1
        self.hidden_units = hidden_units
        self.num_layers = 1
        self.dropout_prob = 0.25

        self.gru = nn.GRU(
            self.input_dim, self.hidden_units, self.num_layers, batch_first=True, dropout=self.dropout_prob
        )

        self.regression = nn.Linear(8, 1)

        self.regression2 = torch.nn.Sequential(
            OrderedDict([
                ('dropout', nn.Dropout(0.25)),
                ('fc1', nn.Linear(7, 1))
            ])
        )

        self.linear = torch.nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_features=self.hidden_units, out_features=32)),
                ('dropout', nn.Dropout(0.25)),
                ('fc2', nn.Linear(32, 14))
            ])
        )

        self.emb_reg = torch.nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_features=384, out_features=32)),
                ('dropout', nn.Dropout(0.25)),
                ('fc2', nn.Linear(32, 1))
            ])
        )
        self.end = nn.Linear(2, 1)

    def forward(self, x, x2, fea):
        bs = x.shape[0]
        h0 = torch.zeros(self.num_layers, bs, self.hidden_units).requires_grad_()
        out, _ = self.gru(x, h0.detach())
        out = self.linear(out[:, -1:, :]).view((bs, 14, 1))
        x2 = self.regression(x2)
        fea = self.regression2(fea).view((bs, 1, 1))
        out = self.end(torch.cat((out, x2), 2))
        with torch.no_grad():
            x3 = (x[:, -14:, :] + torch.cat(
                [(torch.nan_to_num(x[:, -7:, :] - x[:, :7, :], 0, 0) / (x.shape[1] / 7)) * i for i in range(1, 3)], 1)) \
                .view((bs, 14, 1))
        out = x3 + out + fea
        return out


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y, X_req, fea in tqdm(data_loader, total=len(data_loader)):
        output = model(X.float(), X_req.float(), fea.float())
        loss = loss_function(output, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}\n")


def test_model(data_loader, model):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y, X_req, fea in data_loader:
            output = torch.relu(model(X.float(), X_req.float(), fea.float()))
            output = np.exp(output.numpy().reshape(X.size(0), -1)) - 1
            y = np.exp(y.numpy().reshape(X.size(0), -1)) - 1
            total_loss += np.abs(output - y).mean()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}\n")
    return avg_loss


def test_table(data_loader, model):
    outputs = []
    yx = []

    model.eval()
    with torch.no_grad():
        for X, y, X_req, fea in data_loader:
            output = torch.relu(model(X.float(), X_req.float(), fea.float()))
            output = np.exp(output.numpy().reshape(X.size(0), -1)) - 1
            y = np.exp(y.numpy().reshape(X.size(0), -1)) - 1
            outputs.append(output)
            yx.append(y)

    return np.concatenate(outputs), np.concatenate(yx)


def mae_core(error):
    cond = (error < -1).float()
    error = torch.abs((cond * error ** 2) + ((1 - cond) * error))
    loss = torch.mean(error)
    return loss


def MAE(output, target):
    error = output - target
    error2 = output.sum(1) - target.sum(1)
    return mae_core(error) * mae_core(error2)


def bool_map(variable):
    return variable.map({'FALSE': False, 'TRUE': True, False: False, True: True}).fillna(False).astype('int')


if __name__ == "__main__":

    connection_string = 'postgresql://cram@localhost:5432/ordergrid'

    engine = create_engine(connection_string, client_encoding='utf8')

    df = pd.read_sql_query(f"""
        select "createdAt"::date as datetime, warehouse, sku, price, "ageRestricted", "containsAlcohol", "minShelfLife", 
        "VAT", sum(qty) qty from items_datasets
        GROUP BY datetime, warehouse, sku, price, "ageRestricted", "containsAlcohol", "minShelfLife", "VAT" """,
                           con=engine)

    print(f"Dataset loaded with {df.shape[0]} items")

    df = transform_df(df)

    print("Dataset transformed")

    weather_parameters = pd.read_sql_query('select * from weather_parameters', con=engine)

    weather_parameters['datetime'] = pd.to_datetime(weather_parameters['timestamp']).dt.date
    weather_parameters['temp'] = weather_parameters['forecast'].str['temp']
    weather_parameters = weather_parameters.loc[weather_parameters['temp'] > 0, :]
    weather_parameters['clouds_all'] = weather_parameters['forecast'].str['clouds']
    weather_parameters['rain_1h'] = 0  # np.log1p(weather_parameters['rain_1h'].fillna(0))
    weather_parameters['snow_1h'] = 0  # np.log1p(weather_parameters['snow_1h'].fillna(0))
    weather_parameters['humidity'] = weather_parameters['forecast'].str['humidity']
    weather_parameters['wind_speed'] = weather_parameters['forecast'].str['wind_speed']
    weather_parameters['pressure'] = weather_parameters['forecast'].str['pressure']
    weather_parameters = weather_parameters.groupby(['warehouse', 'datetime'], as_index=False) \
        .agg({
        'temp': 'mean',
        'rain_1h': 'sum',
        'snow_1h': 'sum',
        'clouds_all': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'pressure': 'mean'
    })

    print("Weather data is ready")

    df = df.merge(weather_parameters, on=['warehouse', 'datetime'], how='left')
    train_end = str(df['datetime'].max() - pd.to_timedelta(14, 'd'))
    test_start = str(df['datetime'].max() - pd.to_timedelta(41, 'd'))

    print(f"train_end {train_end}")
    print(f"test_start {test_start}")

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    df['temp'] = (df['temp'].fillna(method="ffill") - 273.15) / 30
    df['temp'] = df['temp'].fillna(0.542234)
    df['clouds_all'] = df['clouds_all'].fillna(method="ffill") / 100
    df['clouds_all'] = df['clouds_all'].fillna(0.5)
    df['rain_1h'] = np.log1p(df['rain_1h'].fillna(0))
    df['snow_1h'] = np.log1p(df['snow_1h'].fillna(0))
    df['humidity'] = (df['humidity'].fillna(method="ffill") / 100).fillna(0.77)
    df['wind_speed'] = np.log1p(df['wind_speed'].fillna(method="ffill").fillna(method="bfill"))
    df['pressure'] = (df['pressure'] - 950).fillna(method="ffill").fillna(method="bfill") / 100
    df['value'] = np.log1p(df['value'])
    df = df.fillna(0)

    df_train = df.loc[:train_end].copy()
    df_test = df.loc[test_start:].copy()

    print("Test set fraction:", len(df_test) / len(df))

    train_dataset = SequenceDataset(
        df_train,
    )

    test_dataset = SequenceDataset(
        df_test, train=False
    )

    batch_size = 32 * 8

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y, X_reg, _ = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Features shape:", X_reg.shape)

    learning_rate = 0.01
    num_hidden_units = 64

    model = ForecasterModel(hidden_units=num_hidden_units)
    loss_function = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print("Untrained test\n--------")
    best_score = test_model(test_loader, model)
    print()

    for ix_epoch in range(20):
        print(f"Epoch {ix_epoch}\n---------")
        train_model(train_loader, model, loss_function, optimizer=optimizer)
        score = test_model(test_loader, model)
        if score < best_score:
            print(f"Saving model with new best score {score}\n")
            best_score = score
            torch.save(model.state_dict(), 'models/ts_model.pth') #файл с весами модели
            r, e = test_table(test_loader, model)

            print("Total MAE:", np.abs(r.round() - e).mean())
            print("MAE by days")
            print(pd.DataFrame({
                'day': df_test.index[-14:],
                'MAE': np.abs(r.round() - e).mean(0)
            }))
            print("Total RMSE:", np.sqrt(((r.round() - e) ** 2).mean()))
            print("RMSE by days")
            print(pd.DataFrame({
                'day': df_test.index[-14:],
                'RMSE': np.sqrt(((r.round() - e) ** 2).mean(0))
            }))

            rdf = pd.DataFrame(r)
            rdf.columns = df_test.index[-14:]

            rdf = pd.concat([df_test[['warehouse', 'sku']].drop_duplicates().reset_index(drop=True), rdf], 1)

            rdf = pd.melt(rdf, id_vars=['warehouse', 'sku']).rename(columns={
                'warehouse': 'storeLocationId',
                'variable': 'demandDate',
                'value': 'systemDemand'
            })

            rdf['systemDemand'] = rdf['systemDemand'].round()
            rdf['safetyStock'] = 0

            #rdf.to_csv('nn_model.csv')

        print()
