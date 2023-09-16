import streamlit as st
import pandas as pd
import plotly.io as pio
import plotly.express as px
import LightFM
import numpy as np
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k

st.set_page_config(layout="wide")
pio.templates.default = "plotly_white"

# Add Streamlit title
st.title("Отель - 32 номера")

total_rooms = 32

def load_data():
    try:
        data = pd.read_csv('Radio_history.csv')
    except:
        url = 'https://raw.githubusercontent.com/ngribin/hotel_dash/main/Radio_history.csv'
        data = pd.read_csv(url)
    return data

data = load_data()

st.sidebar.title("Меню")
selected_tab = st.sidebar.radio("Вкладка:", ["Загрузка", "Категории", "Темпы продаж", "Рекомендации"])
# Display data using Streamlit



if selected_tab == 'Загрузка':
    st.subheader('Загрузка')
    st.write("### Бронирования")
    st.write(data)

    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    occupied_rooms = []

    for date in pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])):
        overlapping_bookings = ((data['arrDate'] < date) & (data['depDate'] > date))
        occupied_rooms_count = overlapping_bookings.sum()
        occupied_rooms.append(occupied_rooms_count)

    # Df
    overlap_df = pd.DataFrame({
        'Date': pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])),
        'Occupancy': occupied_rooms
    })

    # Display occupancy chart
    st.write("### Номерной фонд 32 номера")
    st.plotly_chart(px.line(overlap_df, x='Date', y='Occupancy', title="Загрузка - номера").update_layout(
        xaxis_title="Дата", yaxis_title="Загрузка"
    ))

elif selected_tab == "Категории":
    st.subheader("Категории")


    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    data['roomType_id'].unique()

    room_type_mapping = {
        516: 'Номер "Первой" категории двухместный',
        519: 'Номер "Первой" категории одноместный',
        522: 'Номер "Первой" категории двухместный с отдельными кроватями',
        1136: 'Улучшенный номер "Первой" категории двухместный',
        3074: 'Номер "Высшей" категории с круглой ванной',
        3113: 'Номер "Высшей" категории с хаммамом',
        3115: 'Номер "Высшей" категории люкс семейный',
        3116: 'Номер "Высшей" категории люкс',
    }

    unique_room_types = data['roomType_id'].unique()

    # собираем загрузку для каждой категории
    room_type_occupancy = {room_type_mapping[room_type]: [] for room_type in unique_room_types}

    for date in pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])):
        for room_type in unique_room_types:
            overlapping_bookings = ((data['arrDate'] < pd.Timestamp(date)) & (data['depDate'] > pd.Timestamp(date)) & (
                        data['roomType_id'] == room_type))
            occupied_rooms_count = overlapping_bookings.sum()
            room_type_occupancy[room_type_mapping[room_type]].append(occupied_rooms_count)

    occupancy_df = pd.DataFrame({
        'Date': pd.date_range(start=min(data['arrDate']), end=max(data['depDate'])),
        **room_type_occupancy
    })

    # occupancy_df.columns[1:] - катег. с именами
    fig = px.line(occupancy_df, x='Date', y=occupancy_df.columns[1:], title="Загрузка по категориям")

    st.plotly_chart(fig, use_container_width=True)




elif selected_tab == "Темпы продаж":
    st.subheader("Темпы продаж на 3 дня")


    st.sidebar.subheader("Выбрать даты")
    start_date = st.sidebar.date_input("Start Date (Для 3 дней)", pd.Timestamp('2023-07-28'),
                                       key="start_date_tempo")
    end_date = st.sidebar.date_input("End Date (Для 3 дней)", pd.Timestamp('2023-07-30'), key="end_date_tempo")

    # Filter data based on selected dates (convert string to datetime)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Ensure that date columns are Timestamp objects
    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    period_data = data[(data['arrDate'] >= start_date) & (data['arrDate'] <= end_date)]

    booking_totals = {}

    # проходимся по каждому дню
    for date in pd.date_range(start=min(period_data['arrDate']), end=max(period_data['arrDate'])):
        bookings_on_date = period_data[(period_data['arrDate'] <= date) & (period_data['depDate'] >= date)]
        total_bookings = len(bookings_on_date)
        booking_pace_pct = (total_bookings / total_rooms) * 100
        booking_totals[date] = {'Всего бронирований': int(total_bookings), 'Темп продаж': round(booking_pace_pct, 2)}

    d3_booking_pace_df = pd.DataFrame(booking_totals)


    st.write(
        f'Темпы продаж {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}: {booking_pace_pct:.2f}% ')


    #
    st.plotly_chart(
        px.bar(d3_booking_pace_df.T, x=d3_booking_pace_df.T.index, y=['Всего бронирований', 'Темп продаж']).update_layout(xaxis_title="Дата",
                                                                                                   yaxis_title="%"),
        use_container_width=True)
    st.write(d3_booking_pace_df.T)
    # Display the booking pace table
    st.subheader("Темпы продаж на 14")

    start_date = st.date_input("Start Date", pd.Timestamp('2023-07-28'))
    end_date = st.date_input("End Date", pd.Timestamp('2023-08-10'))

    # Filter data based on selected dates (convert string to datetime)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Ensure that date columns are Timestamp objects
    data['arrDate'] = pd.to_datetime(data['arrDate'])
    data['depDate'] = pd.to_datetime(data['depDate'])

    period_data = data[(data['arrDate'] >= start_date) & (data['arrDate'] <= end_date)]
    booking_totals = {}

    # проходимся по каждому дню
    for date in pd.date_range(start=min(period_data['arrDate']), end=max(period_data['arrDate'])):
        bookings_on_date = period_data[(period_data['arrDate'] <= date) & (period_data['depDate'] >= date)]
        total_bookings = len(bookings_on_date)
        booking_pace_pct = (total_bookings / total_rooms) * 100
        booking_totals[date] = {'Всего бронирований': int(total_bookings), 'Темп продаж': round(booking_pace_pct, 2)}

    d3_booking_pace_df = pd.DataFrame(booking_totals)

    st.write(
        f'Темпы продаж {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}: {booking_pace_pct:.2f}% ')

    #
    st.plotly_chart(
        px.bar(d3_booking_pace_df.T, x=d3_booking_pace_df.T.index,
               y=['Всего бронирований', 'Темп продаж']).update_layout(xaxis_title="Дата",
                                                                      yaxis_title="%"),
        use_container_width=True)
    st.write(d3_booking_pace_df.T)

elif selected_tab == "Прогноз":
    st.subheader("На стадии обучения :)")


elif selected_tab == 'Рекомендации':
    st.subheader("Рекомендации")
    try:
        data = pd.read_excel("hotels.xlsx")
    except:
        data = read_excel('https://raw.githubusercontent.com/ngribin/hotel_dash/main/hotels.xlsx')
    data = data.dropna()


    # Create a Streamlit app with multiple tabs
    # st.set_page_config(layout="wide")

    # Function to create recommendations
    def create_recommendations(user_location, user_score, user_stars):
        dataset = Dataset()
        dataset.fit(data['Location'], data['Title'])

        # The interaction matrix
        (interactions, _) = dataset.build_interactions(
            (user, item, 1.0) for user, item in zip(data['Location'], data['Title'])
        )

        # Training the LightFM model
        model = LightFM(loss='warp')
        model.fit(interactions, epochs=30, num_threads=2)

        # Maping user parameters to the dataset's internal IDs
        user_id = dataset.mapping()[0][user_location]

        # Filter hotels based on user_score and user_stars
        filtered_hotels = data[(data['Score'] >= user_score) & (data['Stars'] >= user_stars)]

        # Generate recommendations for the user based on the filtered hotels
        n_users, n_items = interactions.shape
        scores = model.predict(user_id, np.arange(n_items))

        # Get recommendations for the filtered hotels
        # filtered_indices = [dataset.mapping()[2][hotel] for hotel in filtered_hotels['Title']]
        # filtered_scores = scores[filtered_indices]
        # filtered_hotel_indices = np.argsort(-filtered_scores)[:5]
        # filtered_top_hotels = filtered_hotels.iloc[filtered_hotel_indices]['Title']

        filtered_indices = [dataset.mapping()[2][hotel] for hotel in filtered_hotels['Title']]
        filtered_scores = scores[filtered_indices]
        filtered_hotel_indices = np.argsort(-filtered_scores)[:5]
        filtered_top_hotels = filtered_hotels.iloc[filtered_hotel_indices][['Title', 'Url']]

        st.write(
            f"Top 5 hotel recommendations for user in '{user_location}' with a score of at least {user_score} and {user_stars} stars:")
        # for i, hotel in enumerate(filtered_top_hotels, 1):
        #     st.write(f"{i}. {hotel}")
        for i, row in enumerate(filtered_top_hotels.iterrows(), 1):
            index, hotel_data = row
            title = hotel_data['Title']
            url = hotel_data['Url']
            st.write(f"{i}. {title}")
            st.write(f"   URL: {url}")

        st.sidebar.header("User Preferences")
        unique_locations = data['Location'].unique()
        user_location = st.sidebar.selectbox("Location:", unique_locations)
        user_score = st.sidebar.slider("Minimum Score:", 0, 10, 7)
        user_stars = st.sidebar.slider("Minimum Stars:", 0, 5, 4)

        create_recommendations(user_location, user_score, user_stars)
