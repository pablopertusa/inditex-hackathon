import pandas as pd


def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds. Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with. Rounded ot the 2nd decimal.

    If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    The column order and types must be scrictly followed.

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    user_id : int
        Id of the client.

    Returns
    -------
    Pandas Dataframe with some metrics for all the sessions of the given user.
    """

    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])


    # Filtrar el DataFrame por el usuario dado
    user_data = df[df["user_id"] == user_id]


    # Agrupar por sesión y calcular las métricas
    session_metrics = (
        user_data.groupby("session_id")
        .agg(
            total_session_time=("timestamp_local", lambda x: round(x.max().timestamp() - x.min().timestamp(),2)),
            cart_addition_ratio=("add_to_cart", lambda x: x.sum()*100 / len(x))
        )
        .reset_index()
    )
    
    # Añadir la columna user_id
    session_metrics["user_id"] = user_id

    # Reordenar columnas
    session_metrics = session_metrics[["user_id", "session_id", "total_session_time", "cart_addition_ratio"]]

    return session_metrics
