import pandas as pd
import sqlalchemy

import psycopg2 as pg
import pandas.io.sql as psql

# Connector function
def postgres_connector(host, port, database, user, password=None):
    user_info = user if password is None else user + ':' + password
    # example: postgresql://federer:grandestslam@localhost:5432/tennis
    url = 'postgres://%s@%s:%d/%s' % (user_info, host, port, database)
    return sqlalchemy.create_engine(url, client_encoding='utf-8')
# Query
def query_database(query, host='127.0.0.1', port='5005', database="", user="", password=None):
    # Get connect engine
    engine = postgres_connector(
        host,
        port,
        "intern_task",
        "candidate",
        "dcard-data-intern-2020"
    )
    # Query example
    return pd.read_sql(query, engine)