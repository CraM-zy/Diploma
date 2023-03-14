import torch
from sqlalchemy import create_engine
import os
import pandas as pd
import numpy as np
from nn_model import ForecasterModel, transform_df
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings


class SequenceDataset(Dataset):
    def __init__(self, df, train_period=28, test_period=14):
        self.train_period = train_period
        self.test_period = test_period
        self.ds = self.seq_set(df)
        self.legend = pd.DataFrame([_ for _, dt in df.groupby(['warehouse', 'sku'])])
        self.legend.columns = ['warehouse', 'sku']

    def seq_set(self, df):
        ds = []
        self.pairs = df.groupby(['warehouse', 'sku'])
        for _, dt in tqdm(self.pairs, total=df[['warehouse', 'sku']].drop_duplicates().shape[0]):
            ds.append((
                torch.from_numpy(dt['value'].values.reshape(-1, 1)),
                torch.from_numpy(dt[['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'humidity', 'wind_speed', 'pressure', 'weekday']].values[-self.test_period:, :]),
                torch.from_numpy(dt[['price', 'ageRestricted', 'containsAlcohol', 'warehouse_1001',
                                     'warehouse_1002', 'minShelfLife', 'VAT']].iloc[0, :].values)
            ))
        return ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, x_reg, fea = self.ds[i]
        return x, x_reg, fea


def test_table(data_loader, model):
    outputs = []
    model.eval()
    with torch.no_grad():
        for X, X_req, fea in data_loader:
            output = torch.relu(model(X.float(), X_req.float(), fea.float()))
            output = np.exp(output.numpy().reshape(X.size(0), -1)) - 1
            outputs.append(output)
    return np.concatenate(outputs)


def df_stats(df):
    unique_days = df['datetime'].nunique()


if __name__ == "__main__":

    db_name = str(os.environ.get('DB_NAME') or '')
    db_user = str(os.environ.get('DB_USER') or '')
    db_host = str(os.environ.get('DB_HOST') or '')
    db_port = str(os.environ.get('DB_PORT') or '')
    db_pass = str(os.environ.get('DB_PASS') or '')

    #connection_string = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    connection_string = 'postgresql://cram@localhost:5432/ordergrid'
    print("Start model initialization")

    model = ForecasterModel(64)
    model.load_state_dict(torch.load('models/ts_model.pth', encoding='latin1'))
    model.eval()
    print("Model initialized")

    engine = create_engine(connection_string, client_encoding='utf8')
    print("Postgres connected")

    date_threshold = str(
        pd.read_sql_query("""select (max("createdAt")::date - INTERVAL '28 DAY')::date from items_datasets""",
                          con=engine)['date'][0])
    df = pd.read_sql_query(f"""
    select datetime, warehouse, sku, price, "ageRestricted", "containsAlcohol", "minShelfLife", "VAT", sum(qty) qty  
    from (select "createdAt"::date as datetime, * from items_datasets) a 
    where datetime > '{date_threshold}' GROUP BY datetime, warehouse, sku, price, "ageRestricted", "containsAlcohol", 
    "minShelfLife", "VAT" """, con=engine)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.date

    if df['datetime'].nunique() < 28:
        warnings.warn("Not enough days in dataset, we need at least 28 days")
    else:

        print(f"Last day {df['datetime'].max()}")
        print(f"First day {df['datetime'].max() - pd.to_timedelta(28, 'd')}")
        print(f"Dataset loaded with {df.shape[0]} items")

        df = transform_df(df)
        #print(df.groupby(['warehouse', 'sku'], as_index=False)['value'].apply(pd.Series.autocorr))
        print("Data transformed")

        weather_parameters = pd.read_sql_query('select * from weather_parameters', con=engine)
        weather_parameters['datetime'] = pd.to_datetime(weather_parameters['timestamp']).dt.date
        weather_parameters['temp'] = weather_parameters['forecast'].str['temp']
        weather_parameters = weather_parameters.loc[weather_parameters['temp'] > 0, :]
        weather_parameters['clouds_all'] = weather_parameters['forecast'].str['clouds']
        weather_parameters['rain_1h'] = 0 #np.log1p(weather_parameters['rain_1h'].fillna(0))
        weather_parameters['snow_1h'] = 0
        weather_parameters['humidity'] = weather_parameters['forecast'].str['humidity']
        weather_parameters['wind_speed'] = weather_parameters['forecast'].str['wind_speed']
        weather_parameters['pressure'] = weather_parameters['forecast'].str['pressure']
        weather_parameters = weather_parameters.groupby(['warehouse', 'datetime'], as_index=False)\
            .agg({
            'temp': 'mean',
            'rain_1h': 'sum',
            'snow_1h': 'sum',
            'clouds_all': 'mean',
            'humidity': 'mean',
            'wind_speed': 'mean',
            'pressure': 'mean'
        })
        print("Weather data ready to use")

        df = df.merge(weather_parameters, on=['warehouse', 'datetime'], how='left')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        df['temp'] = (df['temp'].fillna(method="ffill").fillna(274) - 273.15) / 30
        df['clouds_all'] = df['clouds_all'].fillna(method="ffill") / 100
        df['clouds_all'] = df['clouds_all'].fillna(0.5)
        df['rain_1h'] = np.log1p(df['rain_1h'].fillna(0))
        df['snow_1h'] = np.log1p(df['snow_1h'].fillna(0))
        df['humidity'] = (df['humidity'].fillna(method="ffill") / 100).fillna(0.77)
        df['wind_speed'] = np.log1p(df['wind_speed'].fillna(method="ffill").fillna(method="bfill"))
        df['pressure'] = (df['pressure'] - 950).fillna(method="ffill").fillna(method="bfill") / 100
        df['value'] = np.log1p(df['value'])

        test_dataset = SequenceDataset(df)
        test_loader = DataLoader(test_dataset, batch_size=32 * 8, shuffle=False)
        print(f"Dataloader ready with {len(test_loader)} batches")

        result = pd.DataFrame(test_table(test_loader, model))
        print(f"Results ready with {result.shape[0]} items")

        safety = np.concatenate([(torch.exp(i[0][-14:]) - 1).numpy().reshape(1, -1) for i in test_dataset.ds], 0)
        safety = pd.DataFrame(((safety - result).abs()**0.25).round())

        result.columns = df.index[-14:] + pd.to_timedelta(14, 'd')
        safety.columns = df.index[-14:] + pd.to_timedelta(14, 'd')

        result = pd.concat([test_dataset.legend, result], 1)
        safety = pd.concat([test_dataset.legend, safety], 1)

        result = pd.melt(result, id_vars=['warehouse', 'sku']).rename(columns={
            'warehouse': 'storeLocationId',
            'variable': 'demandDate',
            'value': 'systemDemand'
        })

        safety = pd.melt(safety, id_vars=['warehouse', 'sku']).rename(columns={
            'warehouse': 'storeLocationId',
            'variable': 'demandDate',
            'value': 'safetyStock'
        })

        result = result.merge(safety, on=['storeLocationId', 'demandDate', 'sku'], how='left')

        result['systemDemand'] = result['systemDemand'].round()
        #print(result)

        result.to_csv('demand_forecast.csv', index=False)
        result.to_sql('demand_forecast', engine, index=False, if_exists='replace')
        print(f"Done!")
