import pandas as pd
import re
from etna.datasets import TSDataset
from etna.models import CatBoostPerSegmentModel
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import MeanTransform
from etna.transforms import SumTransform
from etna.pipeline import Pipeline
from etna.analysis import plot_forecast
import streamlit as st
from etna.metrics import SMAPE


def loading_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        first_column_name = df.columns[0]
        df.rename(columns={first_column_name: 'timestamp'}, inplace=True)
        df = TSDataset.to_dataset(df)
        tsd = TSDataset(df, freq="D")
        return tsd
    except Exception as e:
        st.error(f"Error loading data: {e}, please check explanation")
        return None





st.title('Etna Aramco Inn')
st.subheader('Select your dataset')
uploaded_file = st.file_uploader('', type=["txt", "csv"])
st.write(
    'Make sure that your first column is the time, your second column is the segment and your third column is the target!')
data_loaded = False
if uploaded_file is not None:
    if not data_loaded:
        data_load_state = st.subheader('Loading data...')
        ts = loading_data(uploaded_file)
        data_loaded = True
    try:
        HORIZON = st.text_input("Write the size of the test part:")
        HORIZON = int(HORIZON)
    except ValueError as error:
        st.warning('Please, write only numbers, without any others symbols')
    if ts is not None and HORIZON:
        col1, col2 = st.columns(2)
        show_head = col1.checkbox('Show head of the data')
        go_to_training = col2.checkbox('Go to model training')
        if show_head and go_to_training:
            data_load_state.empty()
            st.subheader('Please, select only one checkbox.')
        elif show_head and not go_to_training:
            st.write(ts)
            data_load_state.empty()
        elif go_to_training:
            data_load_state.empty()
            st.subheader('Choose transformations for your data:')
            available_transforms = {
                "SumTransform": SumTransform(in_column="target", window=12),
                "DateFlagsTransform": DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
                "MeanTransform": MeanTransform(in_column=f"target_lag_{HORIZON}", window=12, seasonality=7),
            }
            selected_transforms = [
            ]
            # Additional checkbox for LagTransform
            if st.checkbox("LagTransform (necessarily)", value=True, key="lag_transform"):
                with st.sidebar:
                    st.subheader("Lag Settings")
                    lag_min = HORIZON + 1
                    lag_max = 200
                    lag_default = HORIZON + 1
                    lag_value = st.slider("Number of Lags", lag_min, lag_max, lag_default)
                lag_transform = LagTransform(in_column="target", lags=list(range(HORIZON, lag_value)),
                                             out_column="target_lag")
                selected_transforms.append(lag_transform)
            # Transforms
            for transform_name, transform_class in available_transforms.items():
                if st.sidebar.checkbox(transform_name):
                    with st.sidebar:
                        st.subheader(f"Parameters for {transform_name}")
                        if transform_name == "SumTransform":
                            window = st.slider(f"Window for {transform_name}", min_value=1, max_value=100, value=12,
                                               key=f"{transform_name}_window")
                            transform_class.window = window
                        elif transform_name == "DateFlagsTransform":
                            week_number_in_month = st.checkbox(f"Week Number in Month for {transform_name}",
                                                               value=True,
                                                               key=f"{transform_name}_week_number_in_month")
                            transform_class.week_number_in_month = week_number_in_month
                        elif transform_name == "MeanTransform":
                            window = st.slider(f"Window for {transform_name}", min_value=1, max_value=100, value=12,
                                               key=f"{transform_name}_window")
                            seasonality = st.slider(f"Seasonality for {transform_name}", min_value=1, max_value=30,
                                                    value=7, key=f"{transform_name}_seasonality")
                            transform_class.window = window
                            transform_class.seasonality = seasonality
                    selected_transforms.append(transform_class)
                elif transform_class in selected_transforms:
                    selected_transforms.remove(transform_class)

            # Button
            if st.button("Apply Transformations and Train the Model"):
                train_ts, test_ts = ts.train_test_split(test_size=HORIZON)
                model = CatBoostPerSegmentModel(logging_level = 'Silent')
                pipeline = Pipeline(model=model, transforms=selected_transforms, horizon=HORIZON)
                model_training_mode = st.subheader('Model is training. Please wait and do not press any other buttons.')
                pipeline.fit(train_ts)
                forecast_ts = pipeline.forecast(test_ts)
                metric = SMAPE(mode="macro")
                metric_value = metric(y_true=test_ts, y_pred=forecast_ts)
                model_training_mode.subheader('The model is successfully trained')
                st.subheader('SMAPE metric value:')
                st.subheader(metric_value)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                fig = plot_forecast(forecast_ts=forecast_ts, test_ts=test_ts, train_ts=train_ts, n_train_samples=100)
                st.pyplot(fig)


        else:
            data_load_state.empty()
            st.write(ts.head(1))

